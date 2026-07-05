from typing import List, Optional

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.layers.pool import global_mean_pool
from gammagl.utils import add_self_loops, degree


def normalize_prompt_type(prompt_type: Optional[str]) -> Optional[str]:
    r"""Normalize the prompt type string to the internal canonical form."""
    if prompt_type is None:
        return None

    prompt_key = str(prompt_type).lower()
    if prompt_key == "edgeprompt":
        return "EdgePrompt"
    if prompt_key in ("edgepromptplus", "edgeprompt_plus", "edgeprompt+"):
        return "EdgePromptplus"
    if prompt_key in ("none", "no_prompt"):
        return None

    raise ValueError("Unsupported prompt type: {}".format(prompt_type))


class EdgePromptGCNConv(tlx.nn.Module):
    r"""A GCN-style convolution layer with optional edge prompt injection.

    Parameters
    ----------
    in_channels: int
        Dimension of the input node features.
    out_channels: int
        Dimension of the output node features.
    name: str, optional
        Name of the module.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            W_init="xavier_uniform",
            b_init=None,
        )
        self.bias = self._get_weights(
            "bias",
            shape=(1, out_channels),
            init=tlx.initializers.zeros(),
        )

    def forward(self, x, edge_index, edge_prompt=None):
        # Add self-loops first so every node can aggregate its own feature.
        num_nodes = int(tlx.get_tensor_shape(x)[0])
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index[0], edge_index[1]
        # Standard symmetric normalization used in GCN.
        deg = degree(col, num_nodes=num_nodes, dtype=tlx.float32)
        deg_inv_sqrt = tlx.pow(deg, -0.5)
        deg_inv_sqrt = tlx.where(
            tlx.is_inf(deg_inv_sqrt),
            tlx.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt,
        )
        norm = tlx.gather(deg_inv_sqrt, row) * tlx.gather(deg_inv_sqrt, col)

        src_x = tlx.gather(x, row)
        if edge_prompt is not None:
            # Inject prompt information into source node features on each edge.
            src_x = src_x + edge_prompt

        messages = self.linear(src_x)
        messages = messages * tlx.expand_dims(norm, axis=-1)
        out = tlx.unsorted_segment_sum(messages, col, num_nodes)

        return out + self.bias


class EdgePrompt(tlx.nn.Module):
    r"""The basic EdgePrompt module with one learnable prompt per layer.

    Parameters
    ----------
    dim_list: list
        Feature dimensions used by each prompt layer.
    name: str, optional
        Name of the module.
    """
    def __init__(self, dim_list: List[int], name: Optional[str] = None):
        super().__init__(name=name)

        self.dim_list = dim_list
        self.global_prompt = []
        for layer, dim in enumerate(dim_list):
            prompt = self._get_weights(
                "global_prompt_{}".format(layer),
                shape=(1, dim),
                init=tlx.initializers.xavier_uniform(),
            )
            self.global_prompt.append(prompt)

    def get_prompt(self, x, edge_index, layer):
        # The vanilla EdgePrompt ignores graph structure and returns a global prompt.
        del x, edge_index
        return self.global_prompt[layer]


class EdgePromptPlus(tlx.nn.Module):
    r"""The EdgePrompt+ module that generates edge-aware prompts dynamically.

    Parameters
    ----------
    dim_list: list
        Feature dimensions used by each prompt layer.
    num_anchors: int
        Number of anchor prompts maintained for each layer.
    name: str, optional
        Name of the module.
    """
    def __init__(
        self,
        dim_list: List[int],
        num_anchors: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.dim_list = dim_list
        self.num_anchors = num_anchors
        self.anchor_prompt = []
        self.projectors = nn.ModuleList()

        for layer, dim in enumerate(dim_list):
            anchor = self._get_weights(
                "anchor_prompt_{}".format(layer),
                shape=(num_anchors, dim),
                init=tlx.initializers.xavier_uniform(),
            )
            self.anchor_prompt.append(anchor)
            self.projectors.append(
                nn.Linear(
                    in_features=2 * dim,
                    out_features=num_anchors,
                    W_init="xavier_uniform",
                )
            )

    def get_prompt(self, x, edge_index, layer):
        # Build an edge representation from source and destination node features,
        # then use it to mix anchor prompts adaptively.
        edge_index, _ = add_self_loops(
            edge_index,
            num_nodes=int(tlx.get_tensor_shape(x)[0]),
        )
        src_x = tlx.gather(x, edge_index[0])
        dst_x = tlx.gather(x, edge_index[1])
        edge_feat = tlx.concat([src_x, dst_x], axis=-1)
        coeff = self.projectors[layer](edge_feat)
        coeff = tlx.leaky_relu(coeff, negative_slope=0.2)
        coeff = tlx.softmax(coeff, axis=-1)
        return tlx.matmul(coeff, self.anchor_prompt[layer])


class EdgePromptGCNModel(tlx.nn.Module):
    r"""A stacked GCN backbone for node or graph representations with EdgePrompt.

    Parameters
    ----------
    feature_dim: int
        Dimension of the input node features.
    hidden_dim: int
        Dimension of hidden representations.
    num_layers: int, optional
        Number of GCN layers.
    drop_rate: float, optional
        Dropout rate applied between hidden layers.
    name: str, optional
        Name of the module.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        drop_rate: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prompt_dims = [feature_dim] + [hidden_dim] * (num_layers - 1)

        self.convs = nn.ModuleList()
        in_dims = self.prompt_dims
        out_dims = [hidden_dim] * num_layers
        for layer, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.convs.append(
                EdgePromptGCNConv(
                    in_dim,
                    out_dim,
                    name="edgeprompt_conv_{}".format(layer),
                )
            )

        self.relu = tlx.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, graph, prompt_type=None, prompt=None, pooling=None):
        # When prompt is enabled, each layer obtains its own edge prompt tensor.
        x, edge_index = graph.x, graph.edge_index
        prompt_type = normalize_prompt_type(prompt_type)

        for layer, conv in enumerate(self.convs):
            edge_prompt = None
            if prompt is not None and prompt_type in ("EdgePrompt", "EdgePromptplus"):
                edge_prompt = prompt.get_prompt(x, edge_index, layer)

            x = conv(x, edge_index, edge_prompt=edge_prompt)
            if layer != self.num_layers - 1:
                x = self.relu(x)
                x = self.dropout(x)

        if pooling == "mean":
            # Graph-level mean pooling for batched graphs.
            batch = getattr(graph, "batch", None)
            if batch is None:
                raise ValueError("Mean pooling requires batched graphs with `batch`.")
            return global_mean_pool(x, batch)

        if pooling == "target":
            # Gather the designated target node from each sampled subgraph.
            if not hasattr(graph, "ptr") or not hasattr(graph, "target_node"):
                raise ValueError(
                    "Target pooling requires batched subgraphs with `ptr` and `target_node`."
                )
            target_index = graph.ptr[:-1] + tlx.reshape(graph.target_node, (-1,))
            return tlx.gather(x, target_index)

        return x


class EdgePromptNodeClassifier(tlx.nn.Module):
    r"""A node classification wrapper built on top of the EdgePrompt GCN backbone.

    Parameters
    ----------
    backbone: EdgePromptGCNModel
        The GCN backbone used to produce node embeddings.
    num_classes: int
        Number of prediction classes.
    prompt_type: str or None
        Prompt strategy, supporting ``EdgePrompt``, ``EdgePromptplus`` or ``None``.
    num_prompts: int, optional
        Number of anchor prompts used by ``EdgePromptplus``.
    name: str, optional
        Name of the module.
    """
    def __init__(
        self,
        backbone: EdgePromptGCNModel,
        num_classes: int,
        prompt_type: Optional[str],
        num_prompts: int = 10,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.backbone = backbone
        self.prompt_type = normalize_prompt_type(prompt_type)

        if self.prompt_type == "EdgePrompt":
            self.prompt = EdgePrompt(backbone.prompt_dims, name="EdgePrompt")
        elif self.prompt_type == "EdgePromptplus":
            self.prompt = EdgePromptPlus(
                backbone.prompt_dims,
                num_anchors=num_prompts,
                name="EdgePromptPlus",
            )
        else:
            self.prompt = None

        self.classifier = nn.Linear(
            in_features=backbone.hidden_dim,
            out_features=num_classes,
            W_init="xavier_uniform",
        )

    def forward(self, graph):
        # The classifier head operates on the prompt-enhanced node embeddings.
        node_emb = self.backbone(
            graph,
            prompt_type=self.prompt_type,
            prompt=self.prompt,
        )
        return self.classifier(node_emb)

    def tuning_weights(self):
        # During prompt tuning, only optimize the classifier and prompt parameters.
        weights = list(self.classifier.trainable_weights)
        if self.prompt is not None:
            weights += list(self.prompt.trainable_weights)
        return weights

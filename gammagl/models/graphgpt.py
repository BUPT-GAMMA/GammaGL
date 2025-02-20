#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union


from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
from torch_geometric.data import Data
import json
import os.path as osp
import glob
from gammagl.models.simple_tokenizer import SimpleTokenizer as _Tokenizer
import tensorlayerx as tlx
from collections import OrderedDict

from gammagl.layers.conv import MessagePassing
from gammagl.mpops import segment_sum
from gammagl.utils import add_self_loops
from gammagl.utils.gfm_utils import DEFAULT_G_END_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

_tokenizer = _Tokenizer("/home/yy3/graph-text-align/GraphGPT/graphgpt/model/graph_layers/bpe_simple_vocab_16e6.txt.gz") # the path of this file, should be found in GraphGPT project


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(tlx.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * tlx.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class GNN(MessagePassing):
    def __init__(self, args, **kwargs):
        super(GNN, self).__init__(aggr='add', **kwargs)
        self.config = PretrainedConfig()
        self.vars = nn.ParameterList()

        w = nn.Parameter(tlx.ones([args.gnn_hid, args.gnn_input]))
        nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_hid)))

        w = nn.Parameter(tlx.ones([args.gnn_output, args.gnn_hid]))
        nn.init.xavier_uniform_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gnn_output)))

    @staticmethod
    def norm(edge_index, num_nodes, improved=False, dtype=None):
        edge_weight = tlx.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = segment_sum(edge_weight, row, num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, g, vars=None):
        device = self.parameters()[0].device
        g = g.to(device)
        
        edge_index = g.edge_index
        x = g.graph_node
        if vars is None:
            vars = self.vars
        improved = False

        w, b = vars[0], vars[1]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        w = w.to(x.device)
        b = b.to(x.device)
        x = F.linear(x, w, b)
        x = F.leaky_relu(x)

        w, b = vars[2], vars[3]
        edge_index, norm = self.norm(edge_index, x.size(self.node_dim), improved, x.dtype)
        x = self.propagate(edge_index, x=x, norm=norm)
        w = w.to(x.device)
        b = b.to(x.device)
        x = F.linear(x, w, b)

        return x

    def parameters(self):
        return self.vars



def Mv2SameDevice(var_list):
    for vid in range(1, len(var_list)):
        var_list[vid] = var_list[vid].to(var_list[0].device)
    return var_list

class CLIP(nn.Module):
    def __init__(self,
                 args
                 ):
        super().__init__()

        self.context_length = args.context_length
        self.args = args
        self.edge_coef = args.edge_coef

        if args.gnn_type == 'gcn':
            self.gnn = GNN(args)
        elif args.gnn_type == 'gt': 
            self.gnn = graph_transformer(args)
        self.transformer = Transformer(
            width=args.transformer_width,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = args.vocab_size
        self.token_embedding = nn.Embedding(args.vocab_size,
                                            args.transformer_width)  # the embedding for all possible tokens
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, args.transformer_width))
        self.ln_final = LayerNorm(args.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(args.transformer_width, args.embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if args.gnn_type == 'gcn':
            self.dtype = self.gnn.vars[0].dtype
        elif args.gnn_type == 'gt': 
            self.dtype = self.gnn.W_pos.dtype

        self.optim = torch.optim.Adam([{'params': self.token_embedding.weight},
                                 {'params': self.positional_embedding},
                                 {'params': self.transformer.parameters()},
                                 {'params': self.text_projection},
                                 {'params': self.gnn.parameters()}
                                 ], lr=args.lr)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = tlx.ops.constant(float("-inf"), shape=[self.context_length, self.context_length])
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, idx_train, g):
        embs = self.gnn(g)
        idx_train = idx_train.to(embs.device)
        idx_train = idx_train
        train_embs = embs[idx_train]
        return train_embs

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0,
                      2)  # NLD -> LND, batch_size * context_length *emb_dim -> context_length * batch_size  *emb_dim
        x = self.transformer(x)
        x = x.permute(1, 0,
                      2)  # LND -> NLD, context_length * batch_size *emb_dim -> batch_size * context_length *emb_dim
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot （end of token） embedding (eot_token is the highest number in each sequence)
        # so there is node need to shorten the context length
        x = x[tlx.arange(x.shape[0]), text.argmax(dim=-1)]  #
        x = x @ self.text_projection
        return x

    def forward(self, g, s_n, t_n, s_n_text, t_n_text, training=True):

        s_image_features = self.encode_image(s_n, g)

        s_text_features = self.encode_text(s_n_text)

        t_text_features = self.encode_text(t_n_text)
        t_text_features = t_text_features.reshape(s_image_features.shape[0], self.args.neigh_num, self.args.gnn_output)
        t_text_features = tlx.mean(t_text_features, dim=1, keepdim=False)
        # normalized features
        s_image_features = s_image_features / s_image_features.norm(dim=-1, keepdim=True)
        s_text_features = s_text_features / s_text_features.norm(dim=-1, keepdim=True)
        t_text_features = t_text_features / t_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits

        labels = tlx.arange(s_image_features.shape[0]).cuda()

        # logit_scale = self.logit_scale.exp()  # the temporature hyperparameter
        # logit_scale, s_image_features, s_text_features = Mv2SameDevice([logit_scale, s_image_features, s_text_features])
        # logits = logit_scale * s_image_features @ s_text_features.t()
        # loss_i = tlx.cross_entropy(logits, labels)
        # loss_t = tlx.cross_entropy(logits.T, labels)
        # node_loss = (loss_i + loss_t) / 2

        # logit_scale, s_image_features, t_text_features = Mv2SameDevice([logit_scale, s_image_features, t_text_features])
        # logits = logit_scale * s_image_features @ t_text_features.t()
        # loss_i = tlx.cross_entropy(logits, labels)
        # loss_t = tlx.cross_entropy(logits.T, labels)
        # gt_loss = (loss_i + loss_t)/2

        # logit_scale, s_text_features, t_text_features = Mv2SameDevice([logit_scale, s_text_features, t_text_features])
        # logits = logit_scale * s_text_features @ t_text_features.t()
        # loss_i = tlx.cross_entropy(logits, labels)
        # loss_t = tlx.cross_entropy(logits.T, labels)
        # tt_loss = (loss_i + loss_t)/2

        

        # shape = [global_batch_size, global_batch_size]
        # return all_loss
        return s_image_features, s_text_features, t_text_features, labels


def tokenize(texts: Union[str, List[str]], context_length: int = 128, truncate: bool = True):

    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=tlx.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = tokens

    return result


class GraphLlamaConfig(LlamaConfig):
    model_type = "GraphLlama"

class GraphPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path): 
    # load conig json
    
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = GraphPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys(): 
        state_dict.pop('logit_scale')
    print('loading graph pre train model')
    model.load_state_dict(state_dict)


    return model, args
def transfer_param_tograph(clip_graph, gnn):
    
    print(clip_graph)
    gnn_state_dict = clip_graph.gnn.state_dict()
    gnn.load_state_dict(gnn_state_dict)
    return gnn


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def pos_encoding(pe, learn_pe, nvar, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((nvar, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((nvar, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((nvar, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(nvar, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class graph_transformer(nn.Module):
    def __init__(self, args):
        super(graph_transformer, self).__init__()
        self.config = PretrainedConfig()
        self.gtLayers = nn.Sequential(*[GTLayer(args) for i in range(args.gt_layers)])

        self.W_pos = pos_encoding('zeros', True, 1, args.att_d_model)
                
        self.W_P = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.inverW_P = nn.Linear(args.att_d_model, args.gnn_output)
        self.args = args

    def forward(self, g):
        # Adj: sp adj
        # x: bs * n * d_model * num_patch
        
        # print(edge_index)
        device = self.parameters().__next__().device
        g = g.to(device)
        
        x = g.graph_node
        
        # x, W_P_weight, W_P_bias= Mv2Samedevice([x, self.W_P.weight, self.W_P.bias])
        # self.W_P.weight = nn.Parameter(W_P_weight.to(x.dtype))
        # self.W_P.bias = nn.Parameter(W_P_bias.to(x.dtype))
        # print(self.W_P.dtype, x.dtype)
        z = self.W_P(x)
        if self.args.if_pos: 
            embeds = self.dropout(z + self.W_pos) 
        else: 
            embeds = self.dropout(z) 
        for gt in self.gtLayers:
            embeds = gt(g, embeds) # bs * num_patch * n * d_model
        # embeds, inverW_P_weight, inverW_P_bias = Mv2Samedevice([embeds, self.inverW_P.weight, self.inverW_P.bias])
        # self.inverW_P.weight = nn.Parameter(inverW_P_weight.to(embeds.dtype))
        # self.inverW_P.bias = nn.Parameter(inverW_P_bias.to(embeds.dtype))
        ret = self.inverW_P(embeds)
        return ret
def Mv2Samedevice(vars): 
    return [var.to(vars[0].device) for var in vars]

class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(torch.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(init(torch.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(init(torch.empty(args.att_d_model, args.att_d_model)))
        if args.att_norm: 
            self.norm = nn.LayerNorm(args.att_d_model, eps=1e-6)
        self.args = args
        
        
    
    def forward(self, g, embeds):
        # Adj: adj
        # x: n * d_model
        rows, cols = g.edge_index
        nvar, _ = embeds.shape
        # print(rows)
        # print(cols)

        rowEmbeds = embeds[rows, :]
        colEmbeds = embeds[cols, :]
        evar, _ = rowEmbeds.shape

        # rowEmbeds, qTrans, kTrans, vTrans = Mv2Samedevice([rowEmbeds, self.qTrans, self.kTrans, self.vTrans])
        # self.qTrans = nn.Parameter(qTrans.to(rowEmbeds.dtype))
        # self.kTrans = nn.Parameter(kTrans.to(rowEmbeds.dtype))
        # self.vTrans = nn.Parameter(vTrans.to(rowEmbeds.dtype))
        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        
        att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = torch.clamp(att, -10.0, 10.0)
        expAtt = torch.exp(att)
        
        tem = torch.zeros([nvar, self.args.head]).to(expAtt.device, dtype=expAtt.dtype)
        # print(tem.device, expAtt.device, rows.device)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8) # bleh
        
        resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([evar, self.args.att_d_model])
        tem = torch.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device, dtype=resEmbeds.dtype)
        rows = rows.to(resEmbeds.device)
        tem = tem.to(resEmbeds.dtype)
        resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm: 
            # resEmbeds, norm_weight, norm_bias = Mv2Samedevice([resEmbeds, self.norm.weight, self.norm.bias])
            # self.norm.weight = nn.Parameter(norm_weight.to(resEmbeds.dtype))
            # self.norm.bias = nn.Parameter(norm_bias.to(resEmbeds.dtype))
            resEmbeds = self.norm(resEmbeds)

        return resEmbeds

class GraphLlamaModel(LlamaModel):
    config_class = GraphLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(GraphLlamaModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            # HACK: for FSDP
            # self.vision_tower = [CLIPVisionModel.from_pretrained(config.graph_tower)]
            # self.arxiv_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)
            clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
            self.graph_tower = graph_transformer(args)
            self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
 
            

            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        if hasattr(config, "use_graph_proj"):
            self.graph_projector = nn.Linear(in_features=config.graph_hidden_size, out_features=config.hidden_size)

    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                                  pretrain_graph_mlp_adapter=None, fsdp=None): # TODO: modify this function
        self.config.graph_tower = graph_tower


        if not hasattr(self, 'graph_tower'):
            clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path) 
            graph_tower = graph_transformer(args)
            graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        else:
            graph_tower = self.graph_tower
        graph_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        if not hasattr(self, 'graph_projector'):
            self.graph_projector = nn.Linear(in_features=self.config.graph_hidden_size, out_features=self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        graph_tower = self.get_graph_tower()
        # graph_data = graph_data[0]
        if graph_tower is not None and (input_ids.shape[1] != 1 or self.training) and graph_data is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            with torch.no_grad():
                if type(graph_data) is list:
                    # variable length images
                    graph_node_features = []
                    if type(graph_data[0]) is Data:
                        for g in graph_data:
                            # print(g)
                            node_forward_out = graph_tower(g)
                            graph_node_features.append(node_forward_out)
                    elif type(graph_data[0]) is dict:
                        for g_dict in graph_data:
                            node_forward_out_1 = graph_tower(g_dict['graph_1'])
                            node_forward_out_2 = graph_tower(g_dict['graph_2'])
                            graph_node_features.append(node_forward_out_1)
                            graph_node_features.append(node_forward_out_2)
                else:
                    raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            if type(graph_data) is list:
                # if type(graph_node_features[0]) is not dict:
                graph_node_features = [self.graph_projector(node_feature) for node_feature in graph_node_features]
                # else: 
                #     graph_node_features = [{'graph_1': self.graph_projector(node_feature['graph_1']), 'graph_2': self.graph_projector(node_feature['graph_2'])} for node_feature in graph_node_features]
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            dummy_graph_features = torch.zeros(256, 128, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_graph_features = self.graph_projector(dummy_graph_features)

            new_input_embeds = []
            cur_graph_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == graph_tower.config.graph_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                if graph_tower.config.use_graph_start_end:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_start_token).sum() != (cur_input_ids == graph_tower.config.graph_end_token).sum():
                        raise ValueError("The number of graph start tokens and graph end tokens should be the same.")
                    graph_start_tokens = (cur_input_ids == graph_tower.config.graph_start_token).nonzero().squeeze(dim=0)
                    # print(graph_start_tokens)
                    for graph_start_token_pos in graph_start_tokens:
                        cur_graph_features = graph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]
                        if cur_input_ids[graph_start_token_pos + num_patches + 1] != graph_tower.config.graph_end_token:
                            raise ValueError("The graph end token should follow the graph start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = tlx.concat((cur_input_embeds[:graph_start_token_pos].detach(), cur_input_embeds[graph_start_token_pos:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:graph_start_token_pos + num_patches + 2], cur_input_embeds[graph_start_token_pos + num_patches + 2:].detach()), axis=0)
                        else:
                            cur_new_input_embeds = tlx.concat((cur_input_embeds[:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:]), axis=0)
                        cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_patch_token).sum() != num_patches:
                        raise ValueError("The number of graph patch tokens should be the same as the number of graph patches.")
                    masked_indices = (cur_input_ids == graph_tower.config.graph_patch_token).nonzero().squeeze(dim=0)
                    mask_index_start = masked_indices[0]
                    if (masked_indices != tlx.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The graph patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = tlx.concat((cur_input_embeds[:mask_index_start].detach(), cur_graph_features, cur_input_embeds[mask_index_start+num_patches:].detach()), axis=0)
                    else:
                        cur_new_input_embeds = tlx.concat((cur_input_embeds[:mask_index_start], cur_graph_features, cur_input_embeds[mask_index_start+num_patches:]), axis=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1

            # print(cur_graph_idx)
            # print(len(graph_node_features))
            assert cur_graph_idx == len(graph_node_features)
            inputs_embeds = tlx.stack(new_input_embeds, axis=0)

        return super(GraphLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class GraphLlamaForCausalLM(LlamaForCausalLM):
    r"""GraphGPT model from the
        `"GraphGPT: Graph Instruction Tuning for Large Language Models"
        <http://arxiv.org/abs/2310.13023>`_ paper. Based on LlamaForCausalLM with graph encoder implemented.

        Parameters
        ----------
        config: GraphLlamaConfig
            Defined as in the config.json file in the model directory
    """
    config_class = GraphLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GraphLlamaModel(config)

        self.lm_head = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def get_vision_tower(self):
        model = self.get_model()
        graph_tower = model.graph_tower
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # graph_node_reps=graph_node_reps, 
            # edge_index_reps=edge_index_reps
            graph_data = graph_data
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = tlx.losses.binary_cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": [kwargs.get("graph_data", None)],
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        return model_inputs

    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                    tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    def forward_no_loss(
        self,
        input_ids = None,
        attention_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # graph_node_reps=graph_node_reps, 
            # edge_index_reps=edge_index_reps
            graph_data = graph_data
        )
        

        return CausalLMOutputWithPast(
            loss=0,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

AutoConfig.register("GraphLlama", GraphLlamaConfig)
AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)

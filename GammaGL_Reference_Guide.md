# GammaGL Library Reference Guide for PyTorch Migration

> **Purpose**: Comprehensive reference document for migrating a PyTorch-based GNN project to GammaGL  
> **Library**: GammaGL (Gamma Graph Library) - A multi-backend Graph Neural Network framework  
> **Backend Support**: PyTorch, PaddlePaddle, MindSpore, TensorFlow, Jittor (via TensorLayerX)  
> **API Style**: Highly similar to PyTorch Geometric (PyG)

---

## Table of Contents

- [Library Overview](#library-overview)
- [Part I: Layers (Operators)](#part-i-layers-operators)
  - [1.1 Convolutional Layers](#11-convolutional-layers)
    - [1.1.1 Message Passing Base Class](#111-message-passing-base-class)
    - [1.1.2 GCNConv](#112-gcnconv)
    - [1.1.3 GATConv](#113-gatconv)
    - [1.1.4 GATV2Conv](#114-gatv2conv)
    - [1.1.5 SAGEConv](#115-sageconv)
    - [1.1.6 GINConv](#116-ginconv)
    - [1.1.7 GCNIIConv](#117-gcnii_conv)
    - [1.1.8 ChebConv](#118-chebconv)
    - [1.1.9 APPNPConv](#119-appnpconv)
    - [1.1.10 AGNNConv](#1110-agnnconv)
    - [1.1.11 EdgeConv](#1111-edgeconv)
    - [1.1.12 SGConv](#1112-sgconv)
    - [1.1.13 GPRConv](#1113-gprconv)
    - [1.1.14 PNAConv](#1114-pnaconv)
    - [1.1.15 RGCNConv](#1115-rgcnconv)
    - [1.1.16 CompConv (CompGCN)](#1116-compconv-compgcn)
    - [1.1.17 HANConv](#1117-hanconv)
    - [1.1.18 HGTConv](#1118-hgtconv)
    - [1.1.19 HeteroConv](#1119-heteroconv)
    - [1.1.20 FILMConv](#1120-filmconv)
    - [1.1.21 GMMConv](#1121-gmmconv)
    - [1.1.22 DNAConv](#1122-dnaconv)
    - [1.1.23 MixHopConv](#1123-mixhopconv)
    - [1.1.24 JumpingKnowledge](#1124-jumpingknowledge)
    - [1.1.25 Additional Conv Layers](#1125-additional-conv-layers)
  - [1.2 Attention Layers](#12-attention-layers)
    - [1.2.1 CentralityEncoding](#121-centralityencoding)
    - [1.2.2 EdgeEncoding](#122-edgeencoding)
    - [1.2.3 SpatialEncoding](#123-spatialencoding)
    - [1.2.4 GraphormerLayer](#124-graphormerlayer)
    - [1.2.5 TransConvLayer](#125-transconvlayer)
    - [1.2.6 GraphConvLayer (SGFormer)](#126-graphconvlayer-sgformer)
    - [1.2.7 HECO Encoder Components](#127-heco-encoder-components)
  - [1.3 Pooling Layers](#13-pooling-layers)
    - [1.3.1 Global Pooling Functions](#131-global-pooling-functions)
    - [1.3.2 global_sort_pool](#132-global_sort_pool)
- [Part II: Models](#part-ii-models)
  - [2.1 Classic GNN Models](#21-classic-gnn-models)
    - [2.1.1 GCNModel](#211-gcnmodel)
    - [2.1.2 GATModel](#212-gatmodel)
    - [2.1.3 GraphSAGE Models](#213-graphsage-models)
    - [2.1.4 GINModel](#214-ginmodel)
    - [2.1.5 ChebNetModel](#215-chebnetmodel)
    - [2.1.6 APPNPModel](#216-appnpmodel)
    - [2.1.7 SGCModel](#217-sgcmodel)
    - [2.1.8 GPRGNNModel](#218-gprgnnmodel)
    - [2.1.9 JKNet](#219-jknet)
    - [2.1.10 MLP](#2110-mlp)
    - [2.1.11 MixHopModel](#2111-mixhopmodel)
    - [2.1.12 PNAModel](#2112-pnamodel)
  - [2.2 Heterogeneous Graph Models](#22-heterogeneous-graph-models)
    - [2.2.1 HAN](#221-han)
    - [2.2.2 HGTModel](#222-hgtmodel)
    - [2.2.3 RGCN](#223-rgcn)
    - [2.2.4 Additional Heterogeneous Models](#224-additional-heterogeneous-models)
  - [2.3 Advanced/Specialized Models](#23-advancedspecialized-models)
    - [2.3.1 Graphormer](#231-graphormer)
    - [2.3.2 DeepWalk / Node2Vec](#232-deepwalk--node2vec)
    - [2.3.3 VGAE](#233-vgae)
    - [2.3.4 DGI / GGD / InfoGraph](#234-dgi--ggd--infograph)
    - [2.3.5 SEAL](#235-seal)
    - [2.3.6 LLM-Integrated Models](#236-llm-integrated-models)
- [Part III: Migration Notes](#part-iii-migration-notes)
  - [3.1 TensorLayerX vs PyTorch Differences](#31-tensorlayerx-vs-pytorch-differences)
  - [3.2 Module Naming Conventions](#32-module-naming-conventions)
  - [3.3 Key Utility Functions](#33-key-utility-functions)

---

## Library Overview

**GammaGL** is a multi-backend GNN library that uses **TensorLayerX (tlx)** as its abstraction layer, supporting PyTorch, PaddlePaddle, MindSpore, TensorFlow, and Jittor as backends. The API design closely mirrors **PyTorch Geometric (PyG)**, making migration relatively straightforward.

### Core Backend Abstraction

| PyTorch/PyG | GammaGL (TensorLayerX) |
|-------------|----------------------|
| `import torch` | `import tensorlayerx as tlx` |
| `torch.nn.Module` | `tlx.nn.Module` |
| `torch.nn.Linear` | `tlx.layers.Linear` |
| `torch.nn.Parameter` | `tlx.nn.Parameter` |
| `torch.nn.Dropout` | `tlx.layers.Dropout` |
| `torch.nn.ReLU` | `tlx.ReLU` / `tlx.nn.ReLU` |
| `torch.nn.ModuleList` | `tlx.nn.ModuleList` |
| `torch.nn.ModuleDict` | `tlx.nn.ModuleDict` |
| `torch.nn.LSTM` | `tlx.nn.LSTM` |
| `torch.nn.LayerNorm` | `tlx.nn.LayerNorm` |
| `torch.nn.BatchNorm1d` | `tlx.nn.BatchNorm1d` |

---

## Part I: Layers (Operators)

### 1.1 Convolutional Layers

Directory: `gammagl/layers/conv/`

#### 1.1.1 Message Passing Base Class

| **File** | `gammagl/layers/conv/message_passing.py` |
|----------|----------------------------------------|
| **Class** | `MessagePassing` |
| **PyG Equivalent** | `torch_geometric.nn.MessagePassing` |

**Functionality**: Base class for all graph convolutional layers. Implements the standard message passing paradigm:

$$\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i, \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right)$$

**Key Methods**:

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `message(x, edge_index, edge_weight=None)` | Constructs messages from source to destination nodes | `x`: node features, `edge_index`: edges [2, E], `edge_weight`: optional edge weights | Message tensor [E, message_dim] |
| `aggregate(msg, edge_index, num_nodes, aggr)` | Aggregates messages to destination nodes | `msg`: messages, `edge_index`: edges, `num_nodes`: node count, `aggr`: 'sum'/'mean'/'max' | Aggregated tensor [N, dim] |
| `message_aggregate(x, edge_index, edge_weight, aggr)` | Fused message + aggregation (uses custom C++/CUDA ops) | Same as message | Aggregated tensor |
| `update(x)` | Updates node embeddings after aggregation | `x`: aggregated messages | Updated node features |
| `propagate(x, edge_index, aggr, **kwargs)` | Orchestrates full message passing pipeline | `x`: features, `edge_index`: edges, `aggr`: aggregation type, `**kwargs`: additional args | Output node features |

**PyTorch Migration Notes**:
- Uses `gammagl.mpops` for optimized segmented operations (`unsorted_segment_sum`, `unsorted_segment_mean`, `unsorted_segment_max`)
- Supports fused `gspmm` kernels when `use_ext` is enabled
- Uses `Inspector` class for dynamic argument distribution (similar to PyG)

---

#### 1.1.2 GCNConv

| **File** | `gammagl/layers/conv/gcn_conv.py` |
|----------|----------------------------------|
| **Class** | `GCNConv` |
| **PyG Equivalent** | `torch_geometric.nn.GCNConv` |

**Paper**: "Semi-supervised Classification with Graph Convolutional Networks" (Kipf & Welling, ICLR 2017)

**Formula**:
$$\mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `norm` | str | Normalization: 'both' (default), 'left', 'right', 'none' |
| `add_bias` | bool | Whether to add learnable bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `edge_weight` [E] (optional), `num_nodes` int
- **Output**: `x` [N, out_channels]

**PyTorch Migration Notes**:
- Applies linear transformation BEFORE message passing (different from some implementations that apply it after)
- Degree normalization computed inline (symmetric: $D^{-1/2} A D^{-1/2}$)

---

#### 1.1.3 GATConv

| **File** | `gammagl/layers/conv/gat_conv.py` |
|----------|----------------------------------|
| **Class** | `GATConv` |
| **PyG Equivalent** | `torch_geometric.nn.GATConv` |

**Paper**: "Graph Attention Networks" (Velickovic et al., ICLR 2018)

**Formula**:
$$\alpha_{i,j} = \frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}[\mathbf{\Theta}\mathbf{x}_i \Vert \mathbf{\Theta}\mathbf{x}_j]\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup \{ i \}} \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}[\mathbf{\Theta}\mathbf{x}_i \Vert \mathbf{\Theta}\mathbf{x}_k]\right)\right)}$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Per-head output feature dimension |
| `heads` | int | Number of attention heads (default: 1) |
| `concat` | bool | Concatenate or average heads (default: True) |
| `negative_slope` | float | LeakyReLU angle (default: 0.2) |
| `dropout_rate` | float | Dropout on attention coefficients (default: 0.0) |
| `add_bias` | bool | Add learnable bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `num_nodes` int
- **Output**: `x` [N, heads*out_channels] (if concat=True) or [N, out_channels] (if concat=False)

**PyTorch Migration Notes**:
- Uses `segment_softmax` from `gammagl.utils` for edge-level softmax (equivalent to PyG's scatter_softmax)
- Supports fused `bspmm` kernel

---

#### 1.1.4 GATV2Conv

| **File** | `gammagl/layers/conv/gatv2_conv.py` |
|----------|------------------------------------|
| **Class** | `GATV2Conv` |
| **PyG Equivalent** | `torch_geometric.nn.GATv2Conv` |

**Paper**: "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022)

**Formula**:
$$\alpha_{i,j} = \frac{\exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}[\mathbf{x}_i \Vert \mathbf{x}_j]\right)\right)}{\sum_{k} \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}[\mathbf{x}_i \Vert \mathbf{x}_k]\right)\right)}$$

**Parameters**: Same as GATConv

| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Per-head output dimension |
| `heads` | int | Number of attention heads (default: 1) |
| `concat` | bool | Concatenate or average heads (default: True) |
| `negative_slope` | float | LeakyReLU angle (default: 0.2) |
| `dropout_rate` | float | Dropout rate (default: 0.0) |
| `add_bias` | bool | Add bias (default: True) |

**Input/Output**: Same as GATConv

**Key Difference from GATConv**: Attention is dynamic (query-dependent) rather than static, allowing any node to attend to any other node. Uses separate `att_src` and `att_dst` weights.

---

#### 1.1.5 SAGEConv

| **File** | `gammagl/layers/conv/sage_conv.py` |
|----------|-----------------------------------|
| **Class** | `SAGEConv` |
| **PyG Equivalent** | `torch_geometric.nn.SAGEConv` |

**Paper**: "Inductive Representation Learning on Large Graphs" (Hamilton et al., NeurIPS 2017)

**Formula**:
$$\mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot \mathrm{mean}_{j \in \mathcal{N}(i)} \mathbf{x}_j$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int/tuple | Input feature dimension (tuple for bipartite) |
| `out_channels` | int | Output feature dimension |
| `activation` | callable | Activation function |
| `aggr` | str | Aggregation: 'mean', 'gcn', 'pool', 'lstm' (default: 'mean') |
| `add_bias` | bool | Add bias (default: True) |

**Input/Output**:
- **Input**: `feat` [N, in_channels] or tuple, `edge` [2, E]
- **Output**: `x` [N, out_channels]

**PyTorch Migration Notes**:
- 'mean' mode: separate `fc_self` and `fc_neigh` linear layers
- 'gcn' mode: uses GCN normalization
- 'pool' mode: applies ReLU(Linear(x)) then max pooling
- 'lstm' mode: uses LSTM aggregator (order-sensitive)

---

#### 1.1.6 GINConv

| **File** | `gammagl/layers/conv/gin_conv.py` |
|----------|----------------------------------|
| **Class** | `GINConv` |
| **PyG Equivalent** | `torch_geometric.nn.GINConv` |

**Paper**: "How Powerful are Graph Neural Networks?" (Xu et al., ICLR 2019)

**Formula**:
$$\mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `nn` | tlx.nn.Module | MLP that maps [in_channels] to [out_channels] |
| `eps` | float | Initial epsilon value (default: 0.0) |
| `train_eps` | bool | Whether epsilon is trainable (default: False) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `size` tuple (optional)
- **Output**: `nn(aggregated)` [N, out_channels]

---

#### 1.1.7 GCNIIConv

| **File** | `gammagl/layers/conv/gcnii_conv.py` |
|----------|------------------------------------|
| **Class** | `GCNIIConv` |
| **PyG Equivalent** | `torch_geometric.nn.GCN2Conv` |

**Paper**: "Simple and Deep Graph Convolutional Networks" (Chen et al., ICML 2020)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `alpha` | float | Initial residual connection strength |
| `beta` | float | Identity mapping strength |
| `variant` | bool | Use GCNII* variant with two separate weight matrices |

**Input/Output**:
- **Input**: `x0` [N, in_channels] (initial features), `x` [N, in_channels] (current), `edge_index`, `edge_weight`, `num_nodes`
- **Output**: `x` [N, out_channels]

---

#### 1.1.8 ChebConv

| **File** | `gammagl/layers/conv/cheb_conv.py` |
|----------|-----------------------------------|
| **Class** | `ChebConv` |
| **PyG Equivalent** | `torch_geometric.nn.ChebConv` |

**Paper**: "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (Defferrard et al., NeurIPS 2016)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `K` | int | Chebyshev filter size (polynomial order) |
| `normalization` | str | Laplacian normalization: 'sym' (default), 'rw', None |

**Input/Output**:
- **Input**: `x` [N, F_in], `edge_index` [2, E], `num_nodes`, `edge_weight` [E] (optional), `lambda_max` (optional)
- **Output**: `x` [N, F_out]

---

#### 1.1.9 APPNPConv

| **File** | `gammagl/layers/conv/appnp_conv.py` |
|----------|------------------------------------|
| **Class** | `APPNPConv` |
| **PyG Equivalent** | `torch_geometric.nn.APPNP` |

**Paper**: "Predict then Propagate: Graph Neural Networks meet Personalized PageRank" (Klicpera et al., ICLR 2019)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `iter_K` | int | Number of propagation iterations |
| `alpha` | float | Teleportation/personalization parameter |
| `drop_rate` | float | Dropout rate |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `edge_weight` [E] (optional), `num_nodes`
- **Output**: `x` [N, out_channels]

---

#### 1.1.10 AGNNConv

| **File** | `gammagl/layers/conv/agnn_conv.py` |
|----------|----------------------------------|
| **Class** | `AGNNConv` |
| **PyG Equivalent** | `torch_geometric.nn.AGNNConv` |

**Paper**: "Attention-based Graph Neural Network for Semi-supervised Learning" (Lee et al., 2018)

**Formula**:
$$P_{i,j} = \frac{\exp(\beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}{\sum_{k} \exp(\beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_k))}$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `require_grad` | bool | Whether beta is trainable (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `num_nodes`
- **Output**: `x` [N, in_channels] (no dimension change)

**Key Feature**: Uses cosine similarity for attention weights (parameter-free propagation matrix).

---

#### 1.1.11 EdgeConv

| **File** | `gammagl/layers/conv/edgeconv.py` |
|----------|----------------------------------|
| **Class** | `EdgeConv` |
| **PyG Equivalent** | `torch_geometric.nn.EdgeConv` |

**Paper**: "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., ACM TOG 2019)

**Formula**:
$$\mathbf{x}^{(k)}_i = \max_{j\in N(i)}h_\Theta(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)} - \mathbf{x}_i^{(k-1)})$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `nn` | tlx.nn.Module | MLP operating on [x_i || x_j - x_i] with input [2 * in_channels] |
| `aggr` | str | Aggregation type: 'max' (default), 'sum', 'mean' |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E]
- **Output**: `x` [N, out_channels] (determined by MLP)

---

#### 1.1.12 SGConv

| **File** | `gammagl/layers/conv/sgc_conv.py` |
|----------|----------------------------------|
| **Class** | `SGConv` |
| **PyG Equivalent** | `torch_geometric.nn.SGConv` |

**Paper**: "Simplifying Graph Convolutional Networks" (Wu et al., ICML 2019)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `iter_K` | int | Number of propagation hops K (default: 2) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `edge_weight` [E] (optional), `num_nodes`
- **Output**: `x` [N, out_channels]

**Key Feature**: No learnable parameters in propagation - only applies linear transformation once, then propagates K times.

---

#### 1.1.13 GPRConv

| **File** | `gammagl/layers/conv/gpr_conv.py` |
|----------|----------------------------------|
| **Class** | `GPRConv` |
| **PyG Equivalent** | `torch_geometric.nn.GPRConv` |

**Paper**: "Adaptive Universal Generalized PageRank Graph Neural Network" (Chien et al., ICLR 2021)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `K` | int | Propagation steps |
| `alpha` | float | Parameter for weight initialization |
| `Init` | str | Initialization method: 'SGC', 'PPR', 'NPPR', 'Random', 'WS' |
| `Gamma` | list | Custom gamma weights (used when Init='WS') |

**Input/Output**:
- **Input**: `x` [N, F], `edge_index` [2, E], `edge_weight` [E] (optional), `num_nodes`
- **Output**: Weighted sum of propagated features [N, F]

---

#### 1.1.14 PNAConv

| **File** | `gammagl/layers/conv/pna_conv.py` |
|----------|----------------------------------|
| **Class** | `PNAConv` |
| **PyG Equivalent** | `torch_geometric.nn.PNAConv` |

**Paper**: "Principal Neighbourhood Aggregation for Graph Nets" (Corso et al., NeurIPS 2020)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `aggregators` | list[str] | ['sum', 'mean', 'min', 'max', 'var', 'std'] |
| `scalers` | list[str] | ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear'] |
| `deg` | tensor | Degree histogram for scaler normalization |
| `edge_dim` | int | Edge feature dimension (optional) |
| `towers` | int | Number of towers (default: 1) |
| `pre_layers` | int | Pre-aggregation MLP layers (default: 1) |
| `post_layers` | int | Post-aggregation MLP layers (default: 1) |
| `divide_input` | bool | Split input across towers (default: False) |

**Input/Output**:
- **Input**: `x` [N, F_in], `edge_index` [2, E], `edge_attr` [E, D] (optional)
- **Output**: `x` [N, F_out]

---

#### 1.1.15 RGCNConv

| **File** | `gammagl/layers/conv/rgcn_conv.py` |
|----------|-----------------------------------|
| **Class** | `RGCNConv` |
| **PyG Equivalent** | `torch_geometric.nn.RGCNConv` |

**Paper**: "Modeling Relational Data with Graph Convolutional Networks" (Schlichtkrull et al., ESWC 2018)

**Formula**:
$$\mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j$$

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int/tuple | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `num_relations` | int | Number of relation types |
| `num_bases` | int | Number of basis decompositions (optional) |
| `num_blocks` | int | Number of block-diagonal blocks (optional) |
| `root_weight` | bool | Add self-loop transformation (default: True) |
| `add_bias` | bool | Add bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `edge_type` [E]
- **Output**: `x` [N, out_channels]

**PyTorch Migration Notes**: Supports both basis-decomposition and block-diagonal-decomposition for parameter efficiency.

---

#### 1.1.16 CompConv (CompGCN)

| **File** | `gammagl/layers/conv/compgcn_conv.py` |
|----------|--------------------------------------|
| **Class** | `CompConv` |
| **PyG Equivalent** | `torch_geometric.nn.CompGCNConv` |

**Paper**: "Composition-based Multi-Relational Graph Convolutional Networks" (Vashishth et al., ICLR 2020)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `num_relations` | int | Number of relations |
| `op` | str | Composition operation: 'sub' (default), 'mult', 'corr' |
| `add_bias` | bool | Add bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `edge_type` [E], `ref_emb` [num_relations, in_channels]
- **Output**: `x` [N, out_channels], `ref_emb` [num_relations, out_channels]

---

#### 1.1.17 HANConv

| **File** | `gammagl/layers/conv/han_conv.py` |
|----------|----------------------------------|
| **Class** | `HANConv` |
| **PyG Equivalent** | `torch_geometric.nn.HANConv` |

**Paper**: "Heterogeneous Graph Attention Network" (Wang et al., WWW 2019)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int/dict | Input dimensions per node type |
| `out_channels` | int | Output feature dimension |
| `metadata` | tuple | (node_types, edge_types) |
| `heads` | int | Number of attention heads (default: 1) |
| `negative_slope` | float | LeakyReLU angle (default: 0.2) |
| `dropout_rate` | float | Dropout rate (default: 0.5) |

**Input/Output**:
- **Input**: `x_dict` {node_type: tensor}, `edge_index_dict` {edge_type: tensor}, `num_nodes_dict` {node_type: int}
- **Output**: `x_dict` {node_type: tensor [N, out_channels*heads]}

**Architecture**: Two-level attention - node-level (GAT per metapath) + semantic-level (attention across metapaths).

---

#### 1.1.18 HGTConv

| **File** | `gammagl/layers/conv/hgt_conv.py` |
|----------|----------------------------------|
| **Class** | `HGTConv` |
| **PyG Equivalent** | `torch_geometric.nn.HGTConv` |

**Paper**: "Heterogeneous Graph Transformer" (Hu et al., WWW 2020)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int/dict | Input dimensions per node type |
| `out_channels` | int | Output feature dimension |
| `metadata` | tuple | (node_types, edge_types) |
| `heads` | int | Number of attention heads (default: 1) |
| `group` | str | Aggregation: 'sum' (default), 'mean', 'min', 'max' |
| `dropout_rate` | float | Dropout rate |

**Input/Output**:
- **Input**: `x_dict` {node_type: tensor}, `edge_index_dict` {edge_type: tensor}
- **Output**: `x_dict` {node_type: tensor [N, out_channels]}

**PyTorch Migration Notes**: Uses type-specific linear projections, relation-specific attention, and residual skip connections with learnable gating.

---

#### 1.1.19 HeteroConv

| **File** | `gammagl/layers/conv/hetero_wrapper.py` |
|----------|---------------------------------------|
| **Class** | `HeteroConv` |
| **PyG Equivalent** | `torch_geometric.nn.HeteroConv` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `convs` | dict | {edge_type: conv_layer} mapping |
| `aggr` | str | Cross-relation aggregation: 'sum' (default), 'mean', 'min', 'max' |

**Input/Output**:
- **Input**: `x_dict` {node_type: tensor}, `edge_index_dict` {edge_type: tensor}
- **Output**: `x_dict` {node_type: tensor}

**Usage Pattern**:
```python
hetero_conv = HeteroConv({
    ('paper', 'cites', 'paper'): GCNConv(64, 16),
    ('author', 'writes', 'paper'): SAGEConv((128, 64), 64),
}, aggr='sum')
out_dict = hetero_conv(x_dict, edge_index_dict)
```

---

#### 1.1.20 FILMConv

| **File** | `gammagl/layers/conv/film_conv.py` |
|----------|-----------------------------------|
| **Class** | `FILMConv` |
| **PyG Equivalent** | `torch_geometric.nn.FiLMConv` |

**Paper**: "GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation" (Marcos et al., 2020)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int/tuple | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `num_relations` | int | Number of relations (default: 1) |
| `act` | callable | Activation function (default: ReLU) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E]
- **Output**: `x` [N, out_channels]

---

#### 1.1.21 GMMConv

| **File** | `gammagl/layers/conv/gmm_conv.py` |
|----------|----------------------------------|
| **Class** | `GMMConv` |
| **PyG Equivalent** | `torch_geometric.nn.GMMConv` |

**Paper**: "Geometric deep learning on graphs and manifolds using mixture model CNNs" (Monti et al., CVPR 2017)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int/tuple | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `dim` | int | Pseudo-coordinate dimension |
| `n_kernels` | int | Number of Gaussian kernels |
| `aggr` | str | Aggregation: 'sum', 'mean', 'max' (default: 'sum') |
| `add_bias` | bool | Add bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `pseudo` [E, dim] (optional)
- **Output**: `x` [N, out_channels]

---

#### 1.1.22 DNAConv

| **File** | `gammagl/layers/conv/dna_conv.py` |
|----------|----------------------------------|
| **Class** | `DNAConv` |
| **PyG Equivalent** | `torch_geometric.nn.DNAConv` |

**Paper**: "Just Jump: Towards Dynamic Neighborhood Aggregation in Graph Neural Networks" (Gasteiger et al., 2019)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `channels` | int | Input/output feature dimension |
| `heads` | int | Number of attention heads (default: 1) |
| `groups` | int | Number of groups for linear projections (default: 1) |
| `dropout` | float | Dropout probability (default: 0.0) |
| `normalize` | bool | Add self-loops + symmetric normalization (default: True) |
| `bias` | bool | Add bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, num_layers, channels] (layer-wise features), `edge_index` [2, E]
- **Output**: `x` [N, channels]

**Key Feature**: Uses multi-head attention to dynamically aggregate from all previous layers.

---

#### 1.1.23 MixHopConv

| **File** | `gammagl/layers/conv/mixhop_conv.py` |
|----------|-------------------------------------|
| **Class** | `MixHopConv` |
| **PyG Equivalent** | `torch_geometric.nn.MixHopConv` |

**Paper**: "MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing" (Abu-El-Haija et al., ICML 2019)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension per hop |
| `p` | list | List of adjacency powers [0, 1, 2, ...] |
| `norm` | str | Normalization: 'both' (default), 'left', 'right', 'none' |
| `add_bias` | bool | Add bias (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `edge_weight` [E] (optional), `num_nodes`
- **Output**: `x` [N, out_channels * len(p)]

---

#### 1.1.24 JumpingKnowledge

| **File** | `gammagl/layers/conv/JumpingKnowledge.py` |
|----------|-----------------------------------------|
| **Class** | `JumpingKnowledge` |
| **PyG Equivalent** | `torch_geometric.nn.JumpingKnowledge` |

**Paper**: "Representation Learning on Graphs with Jumping Knowledge Networks" (Xu et al., ICML 2018)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `mode` | str | Aggregation: 'cat', 'max', 'lstm' |
| `channels` | int | Per-layer channel dimension (required for 'lstm') |
| `num_layers` | int | Number of layers to aggregate (required for 'lstm') |

**Input/Output**:
- **Input**: `xs` list/tensor of [N, channels] from each layer
- **Output**: `x` [N, channels*num_layers] ('cat'), [N, channels] ('max'/'lstm')

---

#### 1.1.25 Additional Conv Layers

| **File** | **Class** | **PyG Equivalent** | **Brief Description** |
|----------|-----------|-------------------|----------------------|
| `fagcn_conv.py` | `FAGCNConv` | - | Feature Augmentation GCN - blends original features with propagated features |
| `fusedgat_conv.py` | `FusedGATConv` | - | Optimized GAT with fused attention kernels |
| `gaan_conv.py` | `GAANConv` | - | Graph Attention Aggregation Network |
| `hardgat_conv.py` | `HardGATConv` | - | Hard attention for neighbor selection |
| `hcha_conv.py` | `HCHAConv` | - | Hypergraph Convolution with Attention |
| `heat_conv.py` | `HEATConv` | - | Heterogeneous Graph Attention with Edge Transformer |
| `hid_conv.py` | `HIDConv` | - | Heterogeneous Information Network convolution |
| `hpn_conv.py` | `HPNConv` | - | Heterogeneous Propagation Network |
| `iehgcn_conv.py` | `IEHGCNConv` | - | Inductive Embedding for Heterogeneous GCN |
| `magcl_conv.py` | `MAGCLConv` | - | Metapath-based Adaptive Graph Contrastive Learning |
| `mgnni_m_iter.py` | `MGNNI_Iter` | - | Multi-GNN with Iterative Refinement |
| `rohehan_conv.py` | `RoheHanConv` | - | Rotated Heterogeneous Attention Network |
| `simplehgn_conv.py` | `SimpleHGNConv` | - | Simplified Heterogeneous Graph Network |
| `dhn_conv.py` | `DHNConv` | - | Deep Heterogeneous Network convolution |

---

### 1.2 Attention Layers

Directory: `gammagl/layers/attention/`

#### 1.2.1 CentralityEncoding

| **File** | `gammagl/layers/attention/centrality_encoder.py` |
|----------|-------------------------------------------------|
| **Class** | `CentralityEncoding` |
| **PyG Equivalent** | Custom (Graphormer-specific) |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `max_in_degree` | int | Maximum in-degree for encoding |
| `max_out_degree` | int | Maximum out-degree for encoding |
| `node_dim` | int | Node feature dimension |

**Input/Output**:
- **Input**: `x` [N, node_dim], `edge_index` [2, E]
- **Output**: `x` [N, node_dim] (with centrality bias added)

**Key Feature**: Learns in-degree and out-degree embeddings, adds them to node features.

---

#### 1.2.2 EdgeEncoding

| **File** | `gammagl/layers/attention/edge_encoder.py` |
|----------|-------------------------------------------|
| **Class** | `EdgeEncoding` |
| **PyG Equivalent** | Custom (Graphormer-specific) |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `edge_dim` | int | Edge feature dimension |
| `max_path_distance` | int | Maximum shortest path distance |

**Input/Output**:
- **Input**: `query` [N, dim], `edge_attr` [E, edge_dim], `edge_paths` list of paths
- **Output**: Edge attention bias matrix [N, N]

---

#### 1.2.3 SpatialEncoding

| **File** | `gammagl/layers/attention/spatial_encoder.py` |
|----------|----------------------------------------------|
| **Class** | `SpatialEncoding` |
| **PyG Equivalent** | Custom (Graphormer-specific) |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `max_path_distance` | int | Maximum shortest path distance for encoding |

**Input/Output**:
- **Input**: `x` [N, dim], `node_paths` (shortest path information)
- **Output**: Spatial bias matrix `b` [N, N]

---

#### 1.2.4 GraphormerLayer

| **File** | `gammagl/layers/attention/graphormer_layer.py` |
|----------|------------------------------------------------|
| **Classes** | `GraphormerAttentionHead`, `GraphormerMultiHeadAttention`, `GraphormerLayer` |
| **PyG Equivalent** | Custom (Graphormer-specific) |

**Parameters (GraphormerLayer)**:
| Name | Type | Description |
|------|------|-------------|
| `node_dim` | int | Node feature dimension |
| `edge_dim` | int | Edge feature dimension |
| `n_heads` | int | Number of attention heads |
| `max_path_distance` | int | Maximum path distance |

**Input/Output**:
- **Input**: `x` [N, node_dim], `edge_attr` [E, edge_dim], `b` spatial bias [N, N], `edge_paths`, `ptr` batch pointer
- **Output**: `x` [N, node_dim]

**Architecture**: Multi-head attention with edge encoding + spatial encoding + feed-forward + LayerNorm (Transformer-style).

---

#### 1.2.5 TransConvLayer

| **File** | `gammagl/layers/attention/sgformer_layer.py` |
|----------|---------------------------------------------|
| **Class** | `TransConvLayer` |
| **PyG Equivalent** | Custom (SGFormer-specific) |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `num_heads` | int | Number of attention heads |
| `use_weight` | bool | Use learnable value projection (default: True) |

**Input/Output**:
- **Input**: `query_input` [N, in_channels], `source_input` [N, in_channels]
- **Output**: `x` [N, out_channels]

---

#### 1.2.6 GraphConvLayer (SGFormer)

| **File** | `gammagl/layers/attention/sgformer_layer.py` |
|----------|---------------------------------------------|
| **Class** | `GraphConvLayer` (extends MessagePassing) |
| **PyG Equivalent** | Custom (SGFormer-specific) |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output feature dimension |
| `use_weight` | bool | Apply linear transformation (default: True) |
| `use_init` | bool | Concatenate initial features (default: False) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `edge_index` [2, E], `x0` [N, in_channels] (if use_init), `num_nodes`
- **Output**: `x` [N, out_channels]

---

#### 1.2.7 HECO Encoder Components

| **File** | `gammagl/layers/attention/heco_encoder.py` |
|----------|-------------------------------------------|
| **Classes** | `metapathSpecificGCN`, `inter_att`, `intra_att`, `Attention`, `Sc_encoder`, `Mp_encoder` |

**Components**:
| Class | Description |
|-------|-------------|
| `metapathSpecificGCN` | GCN for a specific metapath |
| `intra_att` | Intra-view attention within a metapath |
| `inter_att` | Inter-view attention across metapaths |
| `Sc_encoder` | Score encoder for attention |
| `Mp_encoder` | Metapath encoder combining multiple metapath views |

---

### 1.3 Pooling Layers

Directory: `gammagl/layers/pool/`

#### 1.3.1 Global Pooling Functions

| **File** | `gammagl/layers/pool/glob.py` |
|----------|------------------------------|
| **PyG Equivalents** | `torch_geometric.nn.global_*_pool` |

| Function | Formula | Parameters | Input/Output |
|----------|---------|------------|-------------|
| `global_sum_pool(x, batch, size)` | $\sum_{n=1}^{N_i} \mathbf{x}_n$ | `x`: node features, `batch`: batch vector, `size`: batch size | Input: [N, F], Output: [B, F] |
| `global_mean_pool(x, batch, size)` | $\frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n$ | Same as sum pool | Input: [N, F], Output: [B, F] |
| `global_max_pool(x, batch, size)` | $\max_{n=1}^{N_i} \mathbf{x}_n$ | Same as sum pool | Input: [N, F], Output: [B, F] |
| `global_min_pool(x, batch, size)` | $\min_{n=1}^{N_i} \mathbf{x}_n$ | Same as sum pool | Input: [N, F], Output: [B, F] |

**PyTorch Migration Notes**: Direct replacement for PyG's `global_add_pool`, `global_mean_pool`, `global_max_pool`.

---

#### 1.3.2 global_sort_pool

| **File** | `gammagl/layers/pool/glob.py` |
|----------|------------------------------|
| **Function** | `global_sort_pool(x, batch, k)` |
| **PyG Equivalent** | `torch_geometric.nn.global_sort_pool` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `x` | tensor | Node features [N, F] |
| `batch` | tensor | Batch vector [N] |
| `k` | int | Number of nodes to keep per graph |

**Input/Output**:
- **Input**: `x` [N, F], `batch` [N], `k`: int
- **Output**: `x` [B, k*F]

**Key Feature**: Sorts nodes by last feature channel (descending), takes top-k, flattens. Used in DGCNN.

---

## Part II: Models

Directory: `gammagl/models/`

### 2.1 Classic GNN Models

#### 2.1.1 GCNModel

| **File** | `gammagl/models/gcn.py` |
|----------|------------------------|
| **Class** | `GCNModel` |
| **PyG Equivalent** | Typical GCN implementation |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `feature_dim` | int | Input feature dimension |
| `hidden_dim` | int | Hidden layer dimension |
| `num_class` | int | Number of output classes |
| `drop_rate` | float | Dropout rate (default: 0.2) |
| `num_layers` | int | Number of GCN layers (default: 2) |
| `norm` | str | Normalization type (default: 'both') |

**Input/Output**:
- **Input**: `x` [N, feature_dim], `edge_index` [2, E], `edge_weight` [E], `num_nodes`
- **Output**: `x` [N, num_class]

**Architecture**: Linear stack of GCNConv layers with ReLU + Dropout between layers.

---

#### 2.1.2 GATModel

| **File** | `gammagl/models/gat.py` |
|----------|------------------------|
| **Class** | `GATModel` |
| **PyG Equivalent** | Typical GAT implementation |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `feature_dim` | int | Input feature dimension |
| `hidden_dim` | int | Hidden layer dimension (per head) |
| `num_class` | int | Number of output classes |
| `heads` | int | Number of attention heads |
| `drop_rate` | float | Dropout rate |
| `num_layers` | int | Number of GAT layers |

**Input/Output**:
- **Input**: `x` [N, feature_dim], `edge_index` [2, E], `num_nodes`
- **Output**: `x` [N, num_class]

**Architecture**: First layers use concat=True, final layer uses concat=False. ELU activation.

---

#### 2.1.3 GraphSAGE Models

| **File** | `gammagl/models/graphsage.py` |
|----------|------------------------------|
| **Classes** | `GraphSAGE_Full_Model`, `GraphSAGE_Sample_Model` |
| **PyG Equivalents** | Typical GraphSAGE implementations |

**GraphSAGE_Full_Model**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `in_feats` | int | Input feature dimension |
| `n_hidden` | int | Hidden dimension |
| `n_classes` | int | Output classes |
| `n_layers` | int | Number of layers |
| `activation` | callable | Activation function |
| `dropout` | float | Dropout rate |
| `aggregator_type` | str | 'mean', 'gcn', 'pool', 'lstm' |

**GraphSAGE_Sample_Model**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `in_feat` | int | Input feature dimension |
| `hid_feat` | int | Hidden dimension |
| `out_feat` | int | Output classes |
| `drop_rate` | float | Dropout rate |
| `num_layers` | int | Number of layers |

Includes `inference()` method for mini-batch inference on large graphs.

---

#### 2.1.4 GINModel

| **File** | `gammagl/models/gin.py` |
|----------|------------------------|
| **Class** | `GINModel` |
| **PyG Equivalent** | Typical GIN for graph classification |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `hidden_channels` | int | Hidden dimension |
| `out_channels` | int | Output classes |
| `num_layers` | int | Number of GINConv layers (default: 4) |

**Input/Output**:
- **Input**: `x` [N, in_channels] (or None for degree features), `edge_index` [2, E], `batch` [N]
- **Output**: `x` [batch_size, out_channels]

**Architecture**: GINConv layers with MLP -> global_sum_pool -> MLP classifier.

---

#### 2.1.5 ChebNetModel

| **File** | `gammagl/models/chebnet.py` |
|----------|-----------------------------|
| **Class** | `ChebNetModel` |
| **PyG Equivalent** | Typical ChebNet implementation |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `feature_dim` | int | Input feature dimension |
| `hidden_dim` | int | Hidden dimension |
| `out_dim` | int | Output classes |
| `k` | int | Chebyshev filter size |
| `drop_rate` | float | Dropout rate |

**Input/Output**:
- **Input**: `x` [N, F], `edge_index` [2, E], `edge_weight` [E], `num_nodes`
- **Output**: `x` [N, out_dim]

**Key Feature**: Automatically computes `lambda_max` (largest Laplacian eigenvalue) on first forward pass.

---

#### 2.1.6 APPNPModel

| **File** | `gammagl/models/appnp.py` |
|----------|--------------------------|
| **Class** | `APPNPModel` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `out_channels` | int | Output classes |
| `iter_K` | int | Propagation iterations |
| `alpha` | float | Teleportation parameter |
| `drop_rate` | float | Dropout rate |

---

#### 2.1.7 SGCModel

| **File** | `gammagl/models/sgc.py` |
|----------|------------------------|
| **Class** | `SGCModel` |

**Architecture**: Simplified GCN with K propagation steps after single linear transformation.

---

#### 2.1.8 GPRGNNModel

| **File** | `gammagl/models/gprgnn.py` |
|----------|---------------------------|
| **Class** | `GPRGNNModel` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `feature_dim` | int | Input feature dimension |
| `hidden_dim` | int | Hidden dimension |
| `num_class` | int | Output classes |
| `K` | int | Propagation steps |
| `alpha` | float | PPR parameter |
| `Init` | str | Weight initialization method |
| `drop_rate` | float | Dropout rate |

---

#### 2.1.9 JKNet

| **File** | `gammagl/models/jknet.py` |
|----------|--------------------------|
| **Class** | `JKNet` |
| **PyG Equivalent** | `torch_geometric.nn.models.JumpingKnowledge` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `dataset` | object | Dataset with num_node_features, num_classes |
| `mode` | str | JK aggregation: 'max', 'cat', 'lstm' (default: 'max') |
| `num_layers` | int | Number of GCN layers (default: 6) |
| `hidden` | int | Hidden dimension (default: 16) |
| `drop` | float | Dropout rate (default: 0.5) |

**Input/Output**:
- **Input**: `x` [N, F], `edge_index` [2, E], `edge_weight` [E], `num_nodes`
- **Output**: `x` [N, num_classes] (with softmax)

---

#### 2.1.10 MLP

| **File** | `gammagl/models/mlp.py` |
|----------|------------------------|
| **Class** | `MLP` |
| **PyG Equivalent** | Similar to PyG MLP |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `channel_list` | list | List of layer dimensions |
| `in_channels` | int | Input dimension (alternative) |
| `hidden_channels` | int | Hidden dimension (alternative) |
| `out_channels` | int | Output dimension (alternative) |
| `num_layers` | int | Number of layers (alternative) |
| `act` | callable | Activation function (default: LeakyReLU) |
| `act_first` | bool | Apply activation before norm (default: False) |
| `norm` | callable | Normalization layer (default: BatchNorm1d) |
| `dropout` | float | Dropout rate (default: 0.0) |
| `bias` | bool | Use bias in linear layers (default: True) |
| `plain_last` | bool | Last layer has no activation/norm (default: True) |

**Input/Output**:
- **Input**: `x` [N, in_channels], `return_emb` bool (optional)
- **Output**: `x` [N, out_channels], optionally `(x, emb)`

---

#### 2.1.11 MixHopModel

| **File** | `gammagl/models/mixhop.py` |
|----------|---------------------------|
| **Class** | `MixHopModel` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | int | Input feature dimension |
| `hidden_channels` | int | Hidden dimension |
| `num_class` | int | Output classes |
| `p` | list | Adjacency powers |
| `num_layers` | int | Number of MixHop layers |

---

#### 2.1.12 PNAModel

| **File** | `gammagl/models/pna.py` |
|----------|------------------------|
| **Class** | `PNAModel` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `dataset` | object | Dataset with metadata |
| `hidden_channels` | int | Hidden dimension |
| `num_layers` | int | Number of PNA layers |

---

### 2.2 Heterogeneous Graph Models

#### 2.2.1 HAN

| **File** | `gammagl/models/han.py` |
|----------|------------------------|
| **Class** | `HAN` |
| **PyG Equivalent** | `torch_geometric.nn.models.HAN` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `in_channels` | dict | Input dimensions per node type |
| `out_channels` | int | Output classes |
| `metadata` | tuple | (node_types, edge_types) |
| `drop_rate` | float | Dropout rate |
| `hidden_channels` | int | Hidden dimension (default: 128) |
| `heads` | int | Attention heads (default: 8) |

**Input/Output**:
- **Input**: `x_dict` {node_type: tensor}, `edge_index_dict` {edge_type: tensor}, `num_nodes_dict` {node_type: int}
- **Output**: `out_dict` {node_type: tensor [N, out_channels]}

---

#### 2.2.2 HGTModel

| **File** | `gammagl/models/hgt.py` |
|----------|------------------------|
| **Class** | `HGTModel` |
| **PyG Equivalent** | `torch_geometric.nn.models.HGT` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `data` | HeteroGraph | Heterogeneous graph data |
| `hidden_channels` | int | Hidden dimension |
| `out_channels` | int | Output classes |
| `num_heads` | int | Number of attention heads |
| `num_layers` | int | Number of HGT layers |
| `target_node_type` | str | Target node type for prediction |
| `drop_rate` | float | Dropout rate (default: 0.5) |

**Input/Output**:
- **Input**: `x_dict` {node_type: tensor}, `edge_index_dict` {edge_type: tensor}
- **Output**: `x` [N_target, out_channels]

---

#### 2.2.3 RGCN

| **File** | `gammagl/models/rgcn.py` |
|----------|-------------------------|
| **Class** | `RGCN` |
| **PyG Equivalent** | `torch_geometric.nn.models.RGCN` |

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `feature_dim` | int | Input feature dimension |
| `hidden_dim` | int | Hidden dimension |
| `num_class` | int | Output classes |
| `num_relations` | int | Number of relation types |

**Input/Output**:
- **Input**: `edge_index` [2, E], `edge_type` [E]
- **Output**: `x` [N, num_class]

---

#### 2.2.4 Additional Heterogeneous Models

| **File** | **Class** | **Description** |
|----------|-----------|----------------|
| `compgcn.py` | `CompGCN` | Compositional Multi-Relational GCN |
| `fagcn.py` | `FAGCN` | Feature Augmentation GNN |
| `film.py` | `FILMModel` | FiLM-based GNN |
| `heat.py` | `HEATModel` | Heterogeneous Attention with Edge Transformer |
| `heco.py` | `HECO` | Heterogeneous Encoder with Cross-view Attention |
| `herec.py` | `HERec` | Heterogeneous Embedding for Recommendation |
| `iehgcn.py` | `IEHGCN` | Inductive Embedding Heterogeneous GCN |
| `magcl.py` | `MAGCL` | Metapath Adaptive Graph Contrastive Learning |
| `mgnni.py` | `MGNNI` | Multi-GNN with Iterative Refinement |
| `rohehan.py` | `RoheHan` | Rotated Heterogeneous Attention Network |
| `simplehgn.py` | `SimpleHGN` | Simplified Heterogeneous Graph Network |
| `cagcn.py` | `CAGCN` | Context-Aware GCN |

---

### 2.3 Advanced/Specialized Models

#### 2.3.1 Graphormer

| **File** | `gammagl/models/graphormer.py` |
|----------|-------------------------------|
| **Class** | `Graphormer` |
| **PyG Equivalent** | Custom |

**Paper**: "Do Transformers Really Perform Bad for Graph Representation?" (Ying et al., NeurIPS 2021)

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `num_layers` | int | Number of Graphormer layers |
| `input_node_dim` | int | Input node feature dimension |
| `node_dim` | int | Hidden node feature dimension |
| `input_edge_dim` | int | Input edge feature dimension |
| `edge_dim` | int | Hidden edge feature dimension |
| `output_dim` | int | Output dimension |
| `n_heads` | int | Number of attention heads |
| `max_in_degree` | int | Maximum in-degree for encoding |
| `max_out_degree` | int | Maximum out-degree for encoding |
| `max_path_distance` | int | Maximum shortest path distance |

**Input/Output**:
- **Input**: `data` Graph object (with x, edge_index, edge_attr, ptr)
- **Output**: `x` [N, output_dim]

**Key Features**: Centrality encoding, spatial encoding (path-based), edge encoding in attention.

---

#### 2.3.2 DeepWalk / Node2Vec

| **File** | **Class** | **Description** |
|----------|-----------|----------------|
| `deepwalk.py` | `DeepWalk` | Random walk + SkipGram for node embeddings |
| `node2vec.py` | `Node2Vec` | Biased random walk + SkipGram |
| `metapath2vec.py` | `MetaPath2Vec` | Metapath-guided random walks for heterogeneous graphs |
| `skipgram.py` | `SkipGram` | SkipGram model for embedding training |

---

#### 2.3.3 VGAE

| **File** | `gammagl/models/vgae.py` |
|----------|-------------------------|
| **Class** | `VGAE` |
| **PyG Equivalent** | `torch_geometric.nn.models.VGAE` |

**Variational Graph Autoencoder** for link prediction. Uses GCN encoder to produce mean/log-variance for latent space.

---

#### 2.3.4 DGI / GGD / InfoGraph

| **File** | **Class** | **Description** |
|----------|-----------|----------------|
| `dgi.py` | `DGI` | Deep Graph InfoMax - unsupervised via patch-level mutual information |
| `ggd.py` | `GGD` | Graph Gathering Diffusion |
| `infograph.py` | `InfoGraph` | InfoGraph - unsupervised graph-level representation learning |
| `grace.py` | `GRACE` | Graph Contrastive Representation Learning |
| `sp2gcl.py` | `SP2GCL` | Structure-Preserving Graph Contrastive Learning |

---

#### 2.3.5 SEAL

| **File** | `gammagl/models/seal.py` |
|----------|-------------------------|
| **Class** | `SEALModel` |
| **PyG Equivalent** | `torch_geometric.nn.models.SEAL` |

**Paper**: "Link Prediction based on Graph Neural Networks" (Zhang & Chen, NeurIPS 2018)

Extracts enclosing subgraphs around target links and classifies them.

---

#### 2.3.6 LLM-Integrated Models

| **File** | **Class** | **Description** |
|----------|-----------|----------------|
| `graphgpt.py` | `GraphGPT` | Graph instruction tuning for LLMs |
| `llaga.py` | `LLaGA` | LLM-empowered Graph Analytics |
| `simple_tokenizer.py` | `SimpleTokenizer` | Tokenizer for text-graph integration |

---

## Part III: Migration Notes

### 3.1 TensorLayerX vs PyTorch Differences

| Operation | PyTorch | TensorLayerX (GammaGL) |
|-----------|---------|----------------------|
| Tensor creation | `torch.tensor()` | `tlx.convert_to_tensor()` |
| Zeros | `torch.zeros()` | `tlx.zeros()` |
| Ones | `torch.ones()` | `tlx.ones()` |
| Arange | `torch.arange()` | `tlx.arange()` |
| Matmul | `torch.matmul()` | `tlx.matmul()` |
| Gather | `torch.gather()` / indexing | `tlx.gather()` / `tlx.ops.gather()` |
| Concat | `torch.cat()` | `tlx.concat()` |
| Reshape | `tensor.view()` / `tensor.reshape()` | `tlx.reshape()` |
| Softmax | `torch.softmax()` | `tlx.softmax()` |
| Reduction sum | `tensor.sum()` | `tlx.reduce_sum()` |
| Reduction mean | `tensor.mean()` | `tlx.reduce_mean()` |
| Reduction max | `tensor.max()` | `tlx.reduce_max()` |
| Stack | `torch.stack()` | `tlx.stack()` |
| Split | `torch.split()` | `tlx.split()` |
| Transpose | `tensor.T` / `tensor.transpose()` | `tlx.transpose()` |
| Where | `torch.where()` | `tlx.where()` |
| Cast/Type | `tensor.float()` | `tlx.cast()` |
| Device | `tensor.to(device)` | `tensor.device` (automatic) |
| Numpy conversion | `tensor.numpy()` | `tlx.convert_to_numpy()` |
| Backend check | - | `tlx.BACKEND` returns 'torch', 'paddle', etc. |

### 3.2 Module Naming Conventions

| PyTorch/PyG | GammaGL |
|-------------|---------|
| `torch_geometric.nn` | `gammagl.layers` |
| `torch_geometric.nn.conv` | `gammagl.layers.conv` |
| `torch_geometric.nn.pool` | `gammagl.layers.pool` |
| `torch_geometric.data` | `gammagl.data` |
| `torch_geometric.datasets` | `gammagl.datasets` |
| `torch_geometric.loader` | `gammagl.loader` |
| `torch_geometric.utils` | `gammagl.utils` |
| `torch_geometric.transforms` | `gammagl.transforms` |
| `torch_geometric.nn.MessagePassing` | `gammagl.layers.conv.MessagePassing` |
| `torch_scatter.scatter_*` | `gammagl.mpops.unsorted_segment_*` |
| `torch_sparse.SparseTensor` | `gammagl.sparse.SparseAdj` |

### 3.3 Key Utility Functions

| **File** | **Function** | **PyG Equivalent** | **Description** |
|----------|-------------|-------------------|----------------|
| `gammagl/utils/degree.py` | `degree()` | `torch_geometric.utils.degree()` | Compute node degrees |
| `gammagl/utils/loop.py` | `add_self_loops()` | `torch_geometric.utils.add_self_loops()` | Add self-loops to edge_index |
| `gammagl/utils/loop.py` | `remove_self_loops()` | `torch_geometric.utils.remove_self_loops()` | Remove self-loops |
| `gammagl/utils/norm.py` | `calc_gcn_norm()` | `torch_geometric.utils.norm` | Compute GCN normalization weights |
| `gammagl/utils/softmax.py` | `segment_softmax()` | `torch_scatter.scatter_softmax()` | Edge-level softmax per node |
| `gammagl/utils/num_nodes.py` | `maybe_num_nodes()` | `torch_geometric.utils.num_nodes` | Infer number of nodes |
| `gammagl/utils/coalesce.py` | `coalesce()` | `torch_geometric.utils.coalesce()` | Merge duplicate edges |
| `gammagl/utils/subgraph.py` | `k_hop_subgraph()` | `torch_geometric.utils.k_hop_subgraph()` | Extract k-hop subgraph |
| `gammagl/utils/negative_sampling.py` | `negative_sampling()` | `torch_geometric.utils.negative_sampling()` | Sample negative edges |
| `gammagl/utils/shortest_path.py` | `shortest_path_distance()` | - | Compute shortest paths (for Graphormer) |
| `gammagl/utils/to_dense_adj.py` | `to_dense_adj()` | `torch_geometric.utils.to_dense_adj()` | Convert edge_index to dense adjacency |
| `gammagl/utils/to_dense_batch.py` | `to_dense_batch()` | `torch_geometric.utils.to_dense_batch()` | Convert batched features to dense |

---

## Quick Reference: Common Migration Patterns

### Pattern 1: GCN Layer

```python
# PyTorch/PyG
from torch_geometric.nn import GCNConv
conv = GCNConv(in_channels, out_channels)
x = conv(x, edge_index, edge_weight)

# GammaGL
from gammagl.layers.conv import GCNConv
conv = GCNConv(in_channels, out_channels, norm='both')
x = conv(x, edge_index, edge_weight, num_nodes)
```

### Pattern 2: GAT Layer

```python
# PyTorch/PyG
from torch_geometric.nn import GATConv
conv = GATConv(in_channels, out_channels, heads=8)
x = conv(x, edge_index)

# GammaGL
from gammagl.layers.conv import GATConv
conv = GATConv(in_channels, out_channels, heads=8, concat=True)
x = conv(x, edge_index, num_nodes)
```

### Pattern 3: Global Pooling

```python
# PyTorch/PyG
from torch_geometric.nn import global_mean_pool
x = global_mean_pool(x, batch)

# GammaGL
from gammagl.layers.pool import global_mean_pool
x = global_mean_pool(x, batch)
```

### Pattern 4: Heterogeneous Convolution

```python
# PyTorch/PyG
from torch_geometric.nn import HeteroConv, GATConv
conv = HeteroConv({edge_type: GATConv(...) for edge_type in edge_types})

# GammaGL (identical API)
from gammagl.layers.conv import HeteroConv, GATConv
conv = HeteroConv({edge_type: GATConv(...) for edge_type in edge_types})
```

### Pattern 5: Message Passing Custom Layer

```python
# PyTorch/PyG
from torch_geometric.nn import MessagePassing
class MyConv(MessagePassing):
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_j):
        return x_j

# GammaGL (identical API)
from gammagl.layers.conv import MessagePassing
class MyConv(MessagePassing):
    def forward(self, x, edge_index, num_nodes):
        return self.propagate(x, edge_index, num_nodes=num_nodes)
    def message(self, x, edge_index):
        return tlx.gather(x, edge_index[0])
```

---

*This reference document covers all layer and model implementations in the GammaGL library. The API closely mirrors PyTorch Geometric, making migration primarily a matter of replacing import paths and adapting to TensorLayerX tensor operations.*

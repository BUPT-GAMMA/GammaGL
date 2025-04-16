import numpy as np
import tensorlayerx as tlx
# import tensorlayerx.nn as nn
import random
import numba

from gammagl.data import Graph
from gammagl.loader.utils import to_csr
from gammagl.utils.platform_utils import all_to_numpy
from gammagl.layers.conv import GCNConv
from gammagl.models import GCNModel
from gammagl.utils import add_self_loops
from gammagl.utils.num_nodes import maybe_num_nodes
# from gammagl.utils.random_walk_sample import rw_sample, rw_sample_by_edge_index
from gammagl.utils import sort_edge_index


# --------------utils--------------

@numba.njit(cache=True)
def _random_walk_with_edges(node, length, indptr, indices, p=0.0):
    n_id = [numba.int32(-1)] * length  # 初始化节点路径，默认值为 -1
    e_id = [numba.int32(-1)] * (length - 1)  # 初始化边路径，默认值为 -1
    n_id[0] = numba.int32(node)  # 起始节点
    i = numba.int32(1)
    _node = node
    _start = indptr[_node]
    _end = indptr[_node + 1]
    
    while i < length:
        start = indptr[node]
        end = indptr[node + 1]
        
        if start == end:  # 如果当前节点没有邻居，提前结束
            break
        
        sample = random.randint(start, end - 1)
        edge_idx = sample  # 记录边索引
        next_node = indices[sample]
        
        if np.random.uniform(0, 1) > p:
            n_id[i] = next_node
            e_id[i - 1] = edge_idx
        else:
            sample = random.randint(_start, _end - 1)
            edge_idx = sample
            next_node = indices[sample]
            n_id[i] = next_node
            e_id[i - 1] = edge_idx
        
        node = next_node
        i += 1
    
    return np.array(n_id, dtype=np.int32), np.array(e_id, dtype=np.int32)


@numba.njit(cache=True, parallel=True)
def random_walk_parallel_with_edges(start, length, indptr, indices, p):
    n_ids = [np.zeros(length, dtype=np.int32)] * len(start)
    e_ids = [np.zeros(length - 1, dtype=np.int32)] * len(start)
    for i in numba.prange(len(start)):
        n_ids[i], e_ids[i] = _random_walk_with_edges(start[i], length, indptr, indices, p)
    return n_ids, e_ids


def rw_sample_with_edges(graph, start, walk_length, p=0):
    indptr, indices, perm = to_csr(graph, None, False)
    indptr = all_to_numpy(indptr)
    indices = all_to_numpy(indices)
    n_ids, e_ids = random_walk_parallel_with_edges(start, walk_length, indptr, indices, p)
    return n_ids, e_ids


def rw_sample_by_edge_index(edge_index, start, walk_length, p=0):
    graph = Graph(edge_index=edge_index)
    return rw_sample_with_edges(graph, start, walk_length, p)

def to_dense_adj(edge_index, max_num_nodes):  
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵  
    adjacency_matrix = tlx.zeros((max_num_nodes, max_num_nodes), device=edge_index.device)  
      
    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上  
    adjacency_matrix[edge_index[0], edge_index[1]] = 1  
    adjacency_matrix[edge_index[1], edge_index[0]] = 1  
      
    return adjacency_matrix

def compute_E_high(adj_matrix, feat_matrix):
    adj_tensor = tlx.convert_to_tensor(adj_matrix, dtype=tlx.float32)
    feat_tensor = feat_matrix.clone().detach().to(dtype=tlx.float32)

    deg_tensor = adj_tensor.sum(1)
    deg_matrix = tlx.diag(deg_tensor)

    laplacian_tensor = deg_matrix - adj_tensor
    numerator = tlx.matmul(tlx.matmul(feat_tensor.T, laplacian_tensor), feat_tensor)
    denominator = tlx.matmul(feat_tensor.T, feat_tensor)
    
    S_high = numerator.sum() / denominator.sum()

    return S_high.item()

def compute_G_ano(adj_matrix, feat_matrix):
    a_high = compute_E_high(adj_matrix, feat_matrix)
    deg_matrix = tlx.diag(tlx.convert_to_tensor(adj_matrix, dtype=tlx.float32).sum(1))
    s_high = compute_E_high(adj_matrix, deg_matrix)

    return a_high, s_high

def dropout_edge(edge_index, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=tlx.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = tlx.convert_to_tensor(np.random.rand(row.size(0))) >= p
    
    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = tlx.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def dropout_subgraph(edge_index,
                             p: float = 0.2,
                             walks_per_node: int = 1,
                             walk_length: int = 3,
                             num_nodes = None,
                             training: bool = True):
    """
    使用 rw_sample_by_edge_index 实现 dropout_subgraph 功能。
    
    Args:
        edge_index (LongTensor): 图的边索引，形状为 [2, num_edges]。
        p (float, optional): 边采样概率，默认为 0.2。
        walks_per_node (int, optional): 每个节点的随机游走次数，默认为 1。
        walk_length (int, optional): 随机游走长度，默认为 3。
        num_nodes (int, optional): 节点数量。如果为 None，则自动推断。
        training (bool, optional): 是否处于训练模式，默认为 True。
    
    Returns:
        Tuple[LongTensor, BoolTensor]: 更新后的边索引和边掩码。
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Sample probability must be in [0, 1], got {p}")
    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=tlx.bool)

    if not training or p == 0.0:
        return edge_index, edge_mask

    # 推断节点数量
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_orders = None
    ori_edge_index = edge_index
    edge_orders = tlx.arange(0, num_edges)
    edge_index, edge_orders = sort_edge_index(edge_index, edge_orders, num_nodes=num_nodes)

    row, col = edge_index
    sample_mask = tlx.convert_to_tensor(np.random.rand(row.size(0))) <= p
    
    start = row[sample_mask].repeat(walks_per_node)


    # 执行随机游走
    n_id, e_id = rw_sample_by_edge_index(edge_index, start=tlx.convert_to_numpy(start), walk_length=walk_length+1)
    e_id = tlx.convert_to_tensor(e_id)

    if e_id.shape[0] != 0:
        e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
        if edge_orders is not None:
            e_id = edge_orders[e_id]
        edge_mask[e_id] = False
    edge_index = ori_edge_index[:, edge_mask]

    return edge_index, edge_mask


# --------------models--------------
class PreModel(tlx.nn.Module):
    r"""Pre-training Model for an anomaly detector model proposed in `"ADA-GAD: 
        Anomaly-Denoised Autoencoders for Graph Anomaly Detection"
        <https://arxiv.org/abs/2312.14535>`_ paper.

        Parameters
        ----------
        in_dim: int
            Node feature dimension.
        num_hidden: int
            Number of hidden units in each layer.
        encoder_num_layers: int
            Number of layers in the encoder.
        attr_decoder_num_layers: int
            Number of layers in the attribute decoder.
        struct_decoder_num_layers: int
            Number of layers in the struct decoder.
        feat_drop: float
            Dropout rate for GCN layer.
        mask_rate: float, optional
            Rate for masking input features during training. Default: 0.3.
        replace_rate: float, optional
            Rate for replacing masked features with random values. Default: 0.1.
        drop_edge_rate: float, optional
            Rate for randomly dropping edges in the graph. Default: 0.0.
        drop_path_rate: float, optional
            Rate for dropping paths in the graph. Default: 0.0.
        predict_all_edge: float, optional
            Probability of predicting all edges in the graph. Default: 0.
        drop_path_length: int, optional
            Maximum length of paths to drop. Default: 3.
        walks_per_node: int, optional
            Number of random walks per node for structural information extraction. Default: 3.
        select_gano_num: int, optional
            Number of augmentation times, finally select the one of smallest G_ano
        name: str, optional
            Model name.
    """
    def __init__(self,
                in_dim: int,
                num_hidden: int,
                encoder_num_layers: int,
                attr_decoder_num_layers: int,
                struct_decoder_num_layers: int,
                feat_drop: float,
                mask_rate: float = 0.3,
                replace_rate: float = 0.1,
                drop_edge_rate: float = 0.0,
                drop_path_rate: float=0.0,
                predict_all_edge: float=0,
                drop_path_length:int=3,
                walks_per_node:int=3,
                select_gano_num:int=0,
                 name=None):
        super().__init__(name=name)

        self._mask_rate = mask_rate

        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self._drop_edge_rate = drop_edge_rate
        self._drop_path_rate = drop_path_rate

        self.predict_all_edge=predict_all_edge
        self.drop_path_length=drop_path_length

        self.walks_per_node=walks_per_node
        self.neg_s=None

        self.select_gano_num=select_gano_num

        enc_num_hidden = num_hidden

        dec_in_dim = num_hidden
        attr_dec_num_hidden = num_hidden 
        struct_dec_num_hidden = num_hidden 

        self.encoder = GCNModel(feature_dim=in_dim,
                                hidden_dim=enc_num_hidden,
                                num_layers=encoder_num_layers,
                                num_class=enc_num_hidden,
                                drop_rate=feat_drop,
                                )
        
        self.attr_decoder = GCNModel(feature_dim=dec_in_dim,
                                hidden_dim=attr_dec_num_hidden,
                                num_layers=attr_decoder_num_layers,
                                num_class=in_dim,
                                drop_rate=feat_drop,
                                )
        
        self.struct_decoder = GCNModel(feature_dim=dec_in_dim,
                                hidden_dim=struct_dec_num_hidden,
                                num_layers=struct_decoder_num_layers,
                                num_class=in_dim,
                                drop_rate=feat_drop,
                                )
        
        self.enc_mask_token = tlx.nn.Parameter(tlx.zeros((1, in_dim)))
        self.encoder_to_decoder = tlx.layers.Linear(in_features=dec_in_dim, out_features=dec_in_dim, b_init=None)

    def node_denoise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = tlx.convert_to_tensor(np.random.permutation(num_nodes))

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0 and int(self._replace_rate * num_mask_nodes)>0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = tlx.convert_to_tensor(np.random.permutation(num_mask_nodes))

            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = tlx.convert_to_tensor(np.random.permutation(num_nodes))[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def intersection_edge(self,edge_index_1, edge_index_2,max_num_nodes): 
        s1=to_dense_adj(edge_index_1,max_num_nodes=max_num_nodes)
        s2=to_dense_adj(edge_index_2,max_num_nodes=max_num_nodes)
        intersection_s=tlx.minimum(s1,s2)

        intersection_edge_index = intersection_s.nonzero().t()
        # print('intersection_edge_index',intersection_edge_index)

        return intersection_edge_index,intersection_s

    def forward(self, x, edge_index):
        num_nodes=x.size()[0]

        # mask edge to reduce struct uncertainty
        _mask_rate=self._mask_rate
        dence_edge_index=to_dense_adj(edge_index,num_nodes)
        # print("dence_edge_index: ", dence_edge_index)
        # use_e_high=False

        # Node-level denoising pretraining
        if self.select_gano_num:

            G_ano_init=float('inf')
            for j in range(self.select_gano_num):
                use_x, (mask_nodes, keep_nodes) = self.node_denoise(x, _mask_rate)
                a_ano,s_ano = compute_G_ano(dence_edge_index,use_x)
                G_ano = a_ano + s_ano # weight both equal to 1
              #print('G_ano',G_ano)    
                if G_ano < G_ano_init:
                    use_x_select=use_x
                    mask_nodes_select=mask_nodes
                    keep_nodes_select=keep_nodes
                    G_ano_init=G_ano
          #print('final G_ano',G_ano_init)
            use_x=use_x_select
            mask_nodes=mask_nodes_select
            keep_nodes=keep_nodes_select
        else:
            use_x, (mask_nodes, keep_nodes) = self.node_denoise(x, _mask_rate)

        use_x = use_x.to(tlx.float32)
        # print('use_x',use_x.size())
        _drop_path_rate=self._drop_path_rate
        _drop_edge_rate=self._drop_edge_rate

        # mask edge for struct reconstruction


        if _drop_edge_rate > 0:
            # use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
            
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
                    # to_dense_adj(edge_index)[0]
                    a_ano,s_ano=compute_G_ano(to_dense_adj(use_mask_edge_edge_index,max_num_nodes=num_nodes),use_x)
                    G_ano = a_ano + s_ano
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_edge_edge_index_select=use_mask_edge_edge_index
                        masked_edge_edges_select=masked_edge_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_edge_edge_index=use_mask_edge_edge_index_select
                masked_edge_edges=masked_edge_edges_select
            else:
                use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)

            use_mask_edge_edge_index = add_self_loops(use_mask_edge_edge_index)[0]
        else:
            use_mask_edge_edge_index = edge_index


        # mask path for struct reconstruction
        if _drop_path_rate > 0:
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_path_edge_index, masked_path_edges= dropout_subgraph(edge_index, p=_drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node)
                  #print('to_dense_adj(use_mask_path_edge_index)[0],use_x',to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes)[0].size(),use_x.size())
                    a_ano,s_ano=compute_G_ano(to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes),use_x)
                    G_ano = a_ano + s_ano
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_path_edge_index_select=use_mask_path_edge_index
                        masked_path_edges_select=masked_path_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_path_edge_index=use_mask_path_edge_index_select
                masked_path_edges=masked_path_edges_select
            else:
                use_mask_path_edge_index, masked_path_edges= dropout_subgraph(edge_index, p=_drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node)
            
            use_mask_path_edge_index = add_self_loops(use_mask_path_edge_index)[0]
        else:
            use_mask_path_edge_index = edge_index



        # mask edge and path
        use_edge_index,use_s=self.intersection_edge(use_mask_edge_edge_index,use_mask_path_edge_index,num_nodes)

        enc_rep= self.encoder(use_x, use_edge_index, None, None)

        # ---- attribute and edge reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        # loss=0
        # final_mask_rate=0
        # ---- attribute reconstruction ----
        if _mask_rate>0:
            attr_recon = self.attr_decoder(rep, use_edge_index, None, None)
            x_init = x
            x_rec = attr_recon
            # loss += self.criterion(x_rec, x_init)
        # ---- edge reconstruction ----
        if _drop_edge_rate>0 or _drop_path_rate>0 :
            h_recon = self.struct_decoder(rep, use_edge_index, None, None)      
            struct_recon = h_recon @ h_recon.T
            s_init = to_dense_adj(edge_index,max_num_nodes=num_nodes)

            # mask_edge_num=(use_s==0) & (s_init==1)
            # final_mask_rate=mask_edge_num.sum()/edge_index.size()[1]

            if self.predict_all_edge: 
                if  self.neg_s == None:
                    neg_rate = edge_index.size()[1]/(s_init.size()[0]**2)*self.predict_all_edge
                    self.neg_s = tlx.convert_to_tensor(np.random.rand(*tuple(s_init.size()))) <neg_rate
                   
                s_rec = tlx.where((((use_s==0) & (s_init==1))|(self.neg_s).to(use_s.device)),struct_recon,s_init)
            else:
                s_rec = tlx.where((use_s==0) & (s_init==1),struct_recon,s_init)

            # loss += self.criterion(s_rec, s_init)

        
        if _mask_rate == 0:
            return None, None, s_init, s_rec
        
        if _drop_edge_rate == 0 and _drop_path_rate == 0:
            return x_init, x_rec, None, None
        
        # return loss,final_mask_rate
        return x_init, x_rec, s_init, s_rec

class ReModel(tlx.nn.Module):
    r"""Re-training Model for an annomaly detector model proposed in `"ADA-GAD: 
        Anomaly-Denoised Autoencoders for Graph Anomaly Detection"
        <https://arxiv.org/abs/2312.14535>`_ paper.

        Parameters
        ----------
        num_features: int
            Input feature dimension.
        hid_dim: int, optional
            Hidden dimension for all components. Default: 64.
        dropout: float, optional
            Dropout rate for GCN layer. Default: 0.3.
        node_encoder_num_layers: int, optional
            Number of layers in the node encoder. Default: 2.
        edge_encoder_num_layers: int, optional
            Number of layers in the edge encoder. Default: 2.
        subgraph_encoder_num_layers: int, optional
            Number of layers in the subgraph encoder. Default: 2.
        attr_decoder_num_layers: int, optional
            Number of layers in the attribute decoder. Default: 1.
        struct_decoder_num_layers: int, optional
            Number of layers in the structural decoder. Default: 1.
        name: str, optional
            Model name.
    """

    def __init__(self,
                 num_features,
                 hid_dim=64,
                 dropout=0.3,
                 node_encoder_num_layers=2,
                 edge_encoder_num_layers=2,
                 subgraph_encoder_num_layers=2,
                 attr_decoder_num_layers=1,
                 struct_decoder_num_layers=1,
                 name=None):
        super().__init__(name=name)

        self.attr_encoder = GCNModel(feature_dim=num_features,
                            hidden_dim=hid_dim,
                            num_layers=node_encoder_num_layers,
                            num_class=hid_dim,
                            drop_rate=0,
                            )
        
        self.struct_encoder = GCNModel(feature_dim=num_features,
                            hidden_dim=hid_dim,
                            num_layers=edge_encoder_num_layers,
                            num_class=hid_dim,
                            drop_rate=0,
                            )
        
        self.subgraph_encoder = GCNModel(feature_dim=num_features,
                            hidden_dim=hid_dim,
                            num_layers=subgraph_encoder_num_layers,
                            num_class=hid_dim,
                            drop_rate=0,
                            )
        

        self.attention_layer1 = tlx.layers.Linear(in_features=hid_dim*3, out_features=hid_dim*3)
        
        self.attention_layer2 = tlx.nn.Softmax(axis=2)

        decoder_in_dim=hid_dim
        self.attr_decoder = GCNModel(feature_dim=decoder_in_dim,
                            hidden_dim=hid_dim,
                            num_layers=attr_decoder_num_layers,
                            num_class=num_features,
                            drop_rate=dropout,
                            )
        self.struct_decoder = GCNModel(feature_dim=hid_dim,
                            hidden_dim=hid_dim,
                            num_layers=struct_decoder_num_layers,
                            num_class=num_features,
                            drop_rate=dropout,
                            )
        
    def forward(self, x, edge_index):
        h_attr=self.attr_encoder(x, edge_index, None, None)
        h_struct=self.struct_encoder(x,edge_index, None, None)
        h_topology=self.subgraph_encoder(x,edge_index, None, None)

        self.attention=self.attention_layer1(tlx.concat([h_attr,h_struct,h_topology],axis=1))
        self.attention=self.attention_layer2(tlx.reshape(self.attention,(-1,h_attr.size()[-1],3)))

        h=h_attr*self.attention[:,:,0]+h_struct*self.attention[:,:,1]+h_topology*self.attention[:,:,2]
        h=h.to(tlx.float32)

        x_ = self.attr_decoder(h, edge_index, None, None)
        h_ = self.struct_decoder(h, edge_index, None, None)
        s_ = h_ @ h_.T

        return x_, s_


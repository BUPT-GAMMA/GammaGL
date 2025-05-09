
import tensorlayerx as tlx
from tensorlayerx.nn import Module, Linear, LayerNorm, Dropout, Parameter
from typing import Tuple, Optional
from gammagl.layers.conv import MessagePassing

class Graph(dict):
    def __dir__(self):
        return super().__dir__() + list(self.keys())
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError('No such attribute: '+key)
        
    def __setattr__(self, key, value):
        self[key]=value
        
    def copy(self):
        return self.__class__(self)



class EGTConv(MessagePassing):
    
    def _egt(self,
             scale_dot: bool,
             scale_degree: bool,
             num_heads: int,
             dot_dim: int,
             clip_logits_min: float,
             clip_logits_max: float,
             attn_dropout: float,
             attn_maskout: float,
             training: bool,
             num_vns: int,
             QKV,
             G,
             E,
             mask):
        shp = QKV.shape
        QKV = tlx.ops.reshape(QKV, (shp[0], shp[1], -1, num_heads))  # 替代view操作
        Q, K, V = tlx.ops.split(QKV, num_or_size_splits=QKV.shape[2]//dot_dim, axis=2)  # 修改split参数
        A_hat = tlx.einsum('bldh,bmdh->blmh', Q, K)
        
        if self.use_adaptive_sparse:
            A_hat = self._adaptive_sparsity(A_hat)
            
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)

        H_hat = tlx.nn.Ramp(v_min=clip_logits_min, v_max=clip_logits_max)(A_hat) + E
        softmax_layer = tlx.nn.Softmax(axis=2)
        if mask is None:
            if attn_maskout > 0 and training:
                rmask = tlx.zeros_like(H_hat)  
                rmask = tlx.bernoulli(attn_maskout) * -1e9 
                
                gates = tlx.sigmoid(G)#+rmask

                A_tild = softmax_layer(H_hat + rmask) * gates
            else:
                gates = tlx.sigmoid(G)

                A_tild = softmax_layer(H_hat) * gates
        else:
            if attn_maskout > 0 and training:
                rmask = tlx.zeros_like(H_hat)  
                rmask = tlx.bernoulli(attn_maskout) * -1e9 
                
                gates = tlx.sigmoid(G+mask)
                A_tild = softmax_layer(H_hat+mask+rmask) * gates
            else:
                gates = tlx.sigmoid(G+mask)
                A_tild = softmax_layer(H_hat+mask) * gates
        
        if attn_dropout > 0:
            dropout_layer = Dropout(p=attn_dropout)
            A_tild = dropout_layer(A_tild)
            
        V_att = tlx.einsum('blmh,bmkh->blkh', A_tild, V)
        
        if scale_degree:
            degrees = tlx.ops.reduce_sum(gates,axis=2,keepdims=True)
            degree_scalers = tlx.ops.log(1+degrees)
            
            if tlx.BACKEND == 'torch':
                device = degree_scalers.device
                mask = mask.to(device)  

            batch_size, num_nodes = tlx.get_tensor_shape(degrees)[0], tlx.get_tensor_shape(degrees)[1]
            mask = tlx.ops.concat([
                tlx.ones([batch_size, num_vns, 1, 1]), 
                tlx.zeros([batch_size, num_nodes - num_vns, 1, 1])
            ], axis=1)
            mask = tlx.ops.tile(mask, [1, 1, 1, num_heads])  
            
            if tlx.BACKEND == 'torch':
                mask = mask.to(device)
    
            degree_scalers = degree_scalers * (1 - mask) + mask
            V_att = V_att * degree_scalers
        
        V_att = tlx.ops.reshape(V_att,(shp[0],shp[1],num_heads*dot_dim))

        return V_att, H_hat

    def _egt_edge(scale_dot: bool,
                  num_heads: int,
                  dot_dim: int,
                  clip_logits_min: float,
                  clip_logits_max: float,
                  QK,
                  E):
        shp = QK.shape
        Q, K = QK.view(shp[0],shp[1],-1,num_heads).split(dot_dim,dim=2)
        
        A_hat = tlx.einsum('bldh,bmdh->blmh', Q, K)
        
        # 添加边通道的稀疏化
        if self.use_adaptive_sparse:
            A_hat = self._adaptive_sparsity(A_hat)
            
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        
        H_hat = tlx.nn.Ramp(v_min=clip_logits_min, v_max=clip_logits_max)(A_hat) + E
        return H_hat
    
    def __init__(self,
                 node_width                      ,
                 edge_width                      ,
                 num_heads                       ,
                 node_mha_dropout    = 0         ,
                 edge_mha_dropout    = 0         ,
                 node_ffn_dropout    = 0         ,
                 edge_ffn_dropout    = 0         ,
                 attn_dropout        = 0         ,
                 attn_maskout        = 0         ,
                 activation          = 'elu'     ,
                 clip_logits_value   = [-5,5]    ,
                 node_ffn_multiplier = 2.        ,
                 edge_ffn_multiplier = 2.        ,
                 scale_dot           = True      ,
                 scale_degree        = False     ,
                 node_update         = True      ,
                 edge_update         = True      ,
                 use_adaptive_sparse = True      ,
                 sparse_alpha=0.5):
        super().__init__()
        self.node_width          = node_width         
        self.edge_width          = edge_width          
        self.num_heads           = num_heads           
        self.node_mha_dropout    = node_mha_dropout        
        self.edge_mha_dropout    = edge_mha_dropout        
        self.node_ffn_dropout    = node_ffn_dropout        
        self.edge_ffn_dropout    = edge_ffn_dropout        
        self.attn_dropout        = attn_dropout
        self.attn_maskout        = attn_maskout
        self.activation          = activation.upper()      
        self.clip_logits_value   = clip_logits_value   
        self.node_ffn_multiplier = node_ffn_multiplier 
        self.edge_ffn_multiplier = edge_ffn_multiplier 
        self.scale_dot           = scale_dot
        self.scale_degree        = scale_degree        
        self.node_update         = node_update         
        self.edge_update         = edge_update        
        self.use_adaptive_sparse = use_adaptive_sparse
        self.sparse_alpha = sparse_alpha
        assert not (self.node_width % self.num_heads)
        self.dot_dim = self.node_width//self.num_heads
        
        self.mha_ln_h = LayerNorm(normalized_shape=self.node_width)
        self.mha_ln_e = LayerNorm(normalized_shape=self.edge_width)
        self.lin_E = Linear(out_features=num_heads, in_features=edge_width)
        if self.node_update:
            self.lin_QKV = Linear(out_features=self.node_width * 3, in_features=self.node_width)
            self.lin_G = Linear(out_features=self.num_heads, in_features=self.edge_width)
        else:
            self.lin_QKV = Linear(out_features=self.node_width * 2, in_features=self.node_width)
        
        try:
            activation_class = getattr(tlx.nn.activation, self.activation)
            self.ffn_fn = activation_class()
        except AttributeError:
            raise ValueError(f"TensorLayerX激活函数 {self.activation} 不存在")
        if self.node_update:
            self.lin_O_h = Linear(in_features=self.node_width, out_features=self.node_width)
            if self.node_mha_dropout > 0:
                self.mha_drp_h = Dropout(p=self.node_mha_dropout)(self.mha_drp_h)
            
            node_inner_dim  = round(self.node_width*self.node_ffn_multiplier)
            self.ffn_ln_h = LayerNorm(normalized_shape=self.node_width)
            self.lin_W_h_1 = Linear(in_features=self.node_width, out_features=node_inner_dim)
            self.lin_W_h_2 = Linear(in_features=node_inner_dim, out_features=self.node_width)
            if self.node_ffn_dropout > 0:
                self.ffn_drp_h  = Dropout(p=self.node_ffn_dropout)(self.ffn_drp_h)
        
        if self.edge_update:
            self.lin_O_e    = Linear(in_features=self.num_heads, out_features=self.edge_width)
            if self.edge_mha_dropout > 0:
                self.mha_drp_e  = Dropout(p=self.edge_mha_dropout)(self.mha_drp_e)
        
            edge_inner_dim  = round(self.edge_width*self.edge_ffn_multiplier)
            self.ffn_ln_e   = LayerNorm(normalized_shape=self.edge_width)
            self.lin_W_e_1  = Linear(in_features=self.edge_width, out_features=edge_inner_dim)
            self.lin_W_e_2  = Linear(in_features=edge_inner_dim, out_features=self.edge_width)
            if self.edge_ffn_dropout > 0:
                self.ffn_drp_e  = Dropout(p=self.edge_ffn_dropout)(self.ffn_drp_e)
    
    def forward(self, g):
        h, e = g.h, g.e
        mask = g.mask
        
        h_r1 = h
        e_r1 = e
        
        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        
        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)
        
        if self.node_update:
            G = self.lin_G(e_ln)
            V_att, H_hat = self._egt(self.scale_dot,
                                     self.scale_degree,
                                     self.num_heads,
                                     self.dot_dim,
                                     self.clip_logits_value[0],
                                     self.clip_logits_value[1],
                                     self.attn_dropout,
                                     self.attn_maskout,
                                     self.is_train,
                                     0 if 'num_vns' not in g else g.num_vns,
                                     QKV,
                                     G, E, mask)
            
            h = self.lin_O_h(V_att)
            if self.node_mha_dropout > 0:
                h = self.mha_drp_h(h)

            h = tlx.ops.add(h, h_r1)
            
            h_r2 = h
            h_ln = self.ffn_ln_h(h)
            h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
            if self.node_ffn_dropout > 0:
                h = self.ffn_drp_h(h)
            
            h = tlx.ops.add(h, h_r2)
            
        else:
            H_hat = self._egt_edge(self.scale_dot,
                                   self.num_heads,
                                   self.dot_dim,
                                   self.clip_logits_value[0],
                                   self.clip_logits_value[1],
                                   QKV, E)
        
        
        if self.edge_update:
            e = self.lin_O_e(H_hat)
            if self.edge_mha_dropout > 0:
                e = self.mha_drp_e(e)
            
            e = tlx.ops.add(e, e_r1)
            
            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.edge_ffn_dropout > 0:
                e = self.ffn_drp_e(e)
            
            e = tlx.ops.add(e, e_r2)
            
        g = g.copy()
        g.h, g.e = h, e
        return g
    
    def __repr__(self):
        rep = super().__repr__()
        rep = (rep + ' ('
                   + f'num_heads: {self.num_heads},'
                   + f'activation: {self.activation},'
                   + f'attn_maskout: {self.attn_maskout},'
                   + f'attn_dropout: {self.attn_dropout}'
                   +')')
        return rep
    
    def _adaptive_sparsity(self, attn_scores):
        # 动态计算稀疏阈值
        mean_score = tlx.reduce_mean(attn_scores, axis=[1,2], keepdims=True)
        std_score = tlx.reduce_std(attn_scores, axis=[1,2], keepdims=True)
        threshold = mean_score + self.sparse_alpha * std_score
        
        # 生成动态稀疏掩码
        mask = tlx.cast(attn_scores > threshold, attn_scores.dtype)
        return attn_scores * mask



class VirtualNodes(Module):
    def __init__(self, node_width, edge_width, num_virtual_nodes = 1):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_virtual_nodes = num_virtual_nodes

        init = tlx.initializers.RandomNormal(mean=0.0, stddev=1.0)
        
        self.vn_node_embeddings = Parameter(
            data=init(shape=(num_virtual_nodes, self.node_width))
        )
        self.vn_edge_embeddings = Parameter(
            data=init(shape=(num_virtual_nodes, self.edge_width))
        )
    
    def forward(self, g):
        h, e = g.h, g.e
        mask = g.mask
        batch_size = tlx.get_tensor_shape(h)[0]
        node_emb = tlx.ops.tile(
            tlx.ops.expand_dims(self.vn_node_embeddings, 0),
            [batch_size, 1, 1]
        )
        h = tlx.ops.concat([node_emb, h], axis=1)

        e_shape = tlx.get_tensor_shape(e)

        edge_emb_row = tlx.ops.expand_dims(self.vn_edge_embeddings, 1)  # 添加列维度
        edge_emb_col = tlx.ops.expand_dims(self.vn_edge_embeddings, 0)   # 添加行维度
        edge_emb_box = 0.5 * (edge_emb_row + edge_emb_col)
        
        edge_emb_row = tlx.ops.tile(
            tlx.ops.expand_dims(edge_emb_row, 0),
            [e_shape[0], 1, e_shape[2], 1]
        )
        edge_emb_col = tlx.ops.tile(
            tlx.ops.expand_dims(edge_emb_col, 0),
            [e_shape[0], e_shape[1], 1, 1]
        )
        edge_emb_box = tlx.ops.tile(
            tlx.ops.expand_dims(edge_emb_box, 0),
            [e_shape[0], 1, 1, 1]
        )
        
        e = tlx.ops.concat([edge_emb_row, e], axis=1)
        e_col_box = tlx.ops.concat([edge_emb_box, edge_emb_col], axis=1)
        e = tlx.ops.concat([e_col_box, e], axis=2)
        
        g = g.copy()
        g.h, g.e = h, e
        
        g.num_vns = self.num_virtual_nodes
        
        if mask is not None:
            padding = [[0, 0], [self.num_virtual_nodes, 0], [self.num_virtual_nodes, 0], [0, 0]]
            g.mask = tlx.nn.PadLayer(
                padding=padding, 
                mode='CONSTANT',  
                constant_values=0  
            )(mask)
        return g


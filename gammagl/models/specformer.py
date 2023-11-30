import math
import tensorlayerx as tlx

def transpose_qkv(X, num_heads):
    """
    To split the q, k, v with multiheads
    
    Parameters
    ----------
    X: 
       The feature of shape: [bsz, query, embed_dim]
    num_heads:
        The number of heads
    
    Returns
    -------
    Tensor
        The tensor of shape: [bsz, query, num_heads, embed_dim / num_heads]
    """
    X = tlx.reshape(X, (tlx.get_tensor_shape(X)[0],
                        tlx.get_tensor_shape(X)[1], num_heads, -1))
    X = tlx.convert_to_tensor(tlx.convert_to_numpy(X).transpose((0, 2, 1, 3)))
    X = tlx.reshape(X, (-1, tlx.get_tensor_shape(X)[2], tlx.get_tensor_shape(X)[3]))
    return X

def transepose_output(X, num_heads):
    X = tlx.reshape(X, (-1, num_heads, tlx.get_tensor_shape(X)[1],
                        tlx.get_tensor_shape(X)[2]))
    X = tlx.convert_to_tensor(tlx.convert_to_numpy(X).transpose((0, 2, 1, 3)))
    X = tlx.reshape(X, (tlx.get_tensor_shape(X)[0],
                        tlx.get_tensor_shape(X)[1], -1))

    return X

class MultiHeadAttention(tlx.nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = n_heads

        self.W_q = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.relu)
        self.W_k = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.relu)
        self.W_v = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.relu)
        self.W_o = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.relu)
    
    def dot_product_attention(self, querys, keys=None, values=None):
        keys = querys
        values = querys
        d = tlx.get_tensor_shape(querys)[-1]
        scores = tlx.bmm(querys, tlx.transpose(keys, perm=[0, 2, 1])) / math.sqrt(d)
        self.attn_weights = tlx.nn.Softmax(axis=-1)(scores)
        return tlx.bmm(self.attn_weights, values)

    def forward(self, q, k, v):
        is_batched = len(tlx.get_tensor_shape(q)) == 3

        if not is_batched:  # if input with no batchs
            q = tlx.expand_dims(q, axis=0)
            k = tlx.expand_dims(k, axis=0)
            v = tlx.expand_dims(v, axis=0)

        q = transpose_qkv(self.W_q(q), self.num_heads)
        k = transpose_qkv(self.W_k(k), self.num_heads)
        v = transpose_qkv(self.W_v(v), self.num_heads)

        output = self.dot_product_attention(q, k, v)
        output_concat = transepose_output(output, self.num_heads)
        res = self.W_o(output_concat)

        if not is_batched:  # if input with no batchs, remove the batchs which is added at the beginning
            res = tlx.squeeze(res, axis=0)

        return res

class SineEncoding(tlx.nn.Module):
    def __init__(self, hidden_dim=16):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        initor = tlx.nn.initializers.XavierNormal()
        self.eig_w = tlx.nn.Linear(in_features=hidden_dim + 1,
                                   out_features=hidden_dim,
                                   act=tlx.ReLU,
                                   W_init=initor)

    def forward(self, e):
        ee = e * self.constant
        div = tlx.exp(tlx.arange(0, self.hidden_dim, 2, dtype=tlx.float32)
                      * (-math.log(10000) / self.hidden_dim))
        pe = tlx.expand_dims(ee, axis=1) * div
        eeig = tlx.concat(
            (tlx.expand_dims(e, axis=1), tlx.sin(pe), tlx.cos(pe)),
            axis=1
            )
        eeig = tlx.cast(eeig, dtype=tlx.float32)

        return self.eig_w(eeig)

class FeedForwardNetwork(tlx.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = tlx.nn.Linear(in_features=input_dim, out_features=hidden_dim, act=tlx.relu)
        self.layer2 = tlx.nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SpecLayer(tlx.nn.Module):
    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = tlx.nn.Dropout(p=prop_dropout)

        # self.weight with shape: [1, m, d]
        if norm == 'none':
            self.weight = tlx.nn.Parameter(data=tlx.ones((1, nbases, ncombines)))
        else:
            self.weight = tlx.nn.Parameter(tlx.initializers.random_normal(mean=0.0, stddev=0.01) \
                                               (shape=(1, nbases, ncombines)))

        if norm == 'layer':
            self.norm = tlx.nn.LayerNorm(ncombines)
        elif norm == 'batch':
            self.norm = tlx.nn.BatchNorm1d(num_features=ncombines)
        else:
            self.norm = None

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight
        x = tlx.reduce_sum(x, axis=1)

        if self.norm is not None:
            x = self.norm(x)
            x = tlx.relu(x)

        return x

class Specformer(tlx.nn.Module):

    r"""The Specformer from the `"Specformer:Spectral Graph Neural Networks Meet Transformers"
    <https://openreview.net/pdf?id=0pdSt3oyJa1>`_ paper

    Parameters
    ----------
    nclass: int
        the number of node classes
    n_feat: int
        the node feature input dimension
    n_layer: int
        number of Speclayers
    hidden_dim: int
        the eigvalue representation dimension and the node feature dimension
    n_heads: int
        the number of attention heads
    tran_dropout: int
        the probability of dropout
    feat_dropout: int
        the probability of dropout
    prop_dropout: float
        the probability of dropout
    """


    def __init__(self, nclass, n_feat, n_layer=2, hidden_dim=32, n_heads=4,
                 tran_dropout=0.2, feat_dropout=0.4, prop_dropout=0.5, norm='none'):
        super(Specformer, self).__init__()

        # node feat: n_feat, eigenvalue repre shape: hidden_dim
        self.norm = norm
        self.n_feat = n_feat
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.feat_encoder = tlx.nn.Sequential(
            tlx.nn.Linear(in_features=n_feat, out_features=hidden_dim, act=tlx.relu),
            tlx.nn.Linear(in_features=hidden_dim, out_features=nclass),
        )
        self.linear_encoder = tlx.nn.Linear(in_features=n_feat, out_features=hidden_dim, act=tlx.relu)
        self.classify = tlx.nn.Linear(in_features=hidden_dim, out_features=nclass, act=tlx.relu)
        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = tlx.nn.Linear(in_features=hidden_dim, out_features=n_heads, act=tlx.relu)
        self.mha_norm = tlx.nn.LayerNorm(hidden_dim)
        self.ffn_norm = tlx.nn.LayerNorm(hidden_dim)
        self.mha_dropout = tlx.nn.Dropout(p=tran_dropout)
        self.ffn_dropout = tlx.nn.Dropout(p=tran_dropout)
        self.mha = MultiHeadAttention(hidden_dim=hidden_dim, n_heads=n_heads, tran_dropout=tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
        self.feat_dp1 = tlx.nn.Dropout(p=feat_dropout)
        self.feat_dp2 = tlx.nn.Dropout(p=feat_dropout)
        
        list_layer = []
        if norm == "none":
            for _ in range(n_layer):
                list_layer.append(SpecLayer(n_heads+1, nclass, prop_dropout, norm=norm))
        else:
            for _ in range(n_layer):
                list_layer.append(SpecLayer(n_heads+1, hidden_dim, prop_dropout, norm=norm)) 

        self.layers = list_layer


    def forward(self, x, edge_index, e, u):
        ut = tlx.transpose(u, perm=[1, 0])
        x = tlx.cast(x=x, dtype=tlx.float32)
        ut = tlx.cast(x=ut, dtype=tlx.float32)
        e = tlx.cast(x=e, dtype=tlx.float32)
        u = tlx.cast(x=u, dtype=tlx.float32)

        # make conversion of node feat
        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = tlx.cast(x=h, dtype=tlx.float32)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        # conversion of eigenvalues repre
        eig = self.eig_encoder(e)

        mha_eig = self.mha_norm(eig)
        mha_eig = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)
        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)
        new_e = self.decoder(eig)

        for i in range(self.n_layer):
            basic_feats = [h]
            utx = ut @ h
            for j in range(self.n_heads):
                basic_feats.append(u @ (tlx.expand_dims(input=new_e[:, j], axis=1) * utx))
            basic_feats = tlx.stack(values=basic_feats, axis=1)

            h = self.layers[i](basic_feats)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h

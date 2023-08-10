import math
from numpy.linalg import eigh
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import WithLoss



def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)  # np.sum
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum
def normalize_graph(g):  # g是邻接矩阵，函数返回L矩阵。函数只使用np，不需要torch
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L  # L = In − D^−1/2 @ A @ D^−1/2,
def eigen_decompositon(g):  # 输入一个邻接矩阵g，返回L矩阵的特征值和特征向量
    "The normalized (unit “length”) eigenvectors, "
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    g = normalize_graph(g)  # 返回邻接矩阵g的L矩阵
    e, u = eigh(g)  # 返回L矩阵的特征值和特征向量
    return e, u
def transpose_qkv_tlx(X, num_heads):
    """
    用于将输入的q,k,v按照多个头进行拆分
    :param X: [bsz,query数量，embed_dim]
    :param num_heads:
    :return: [bsz,query数量，num_heads,embed_dim/num_heads]
    """
    X = tlx.ops.reshape(X,
                        (tlx.ops.get_tensor_shape(X)[0], tlx.ops.get_tensor_shape(X)[1], num_heads, -1))
    X = tlx.ops.convert_to_tensor(
        tlx.ops.convert_to_numpy(X).transpose((0, 2, 1, 3))  # 把num_heads移动到query数量维前面
    )
    X = tlx.ops.reshape(
        X, (-1, tlx.ops.get_tensor_shape(X)[2], tlx.ops.get_tensor_shape(X)[3])
    )
    return X
def transepose_output_tlx(X, num_heads):
    X = tlx.ops.reshape(
        X, (-1, num_heads, tlx.ops.get_tensor_shape(X)[1], tlx.ops.get_tensor_shape(X)[2])
    )
    X = tlx.ops.convert_to_tensor(
        tlx.ops.convert_to_numpy(X).transpose((0, 2, 1, 3))  # 把num_heads移动到query数量维前面
    )
    X = tlx.ops.reshape(
        X, (tlx.ops.get_tensor_shape(X)[0], tlx.ops.get_tensor_shape(X)[1], -1)
    )

    return X
def get_split(y, nclass, seed=0):

    percls_trn = int(round(0.6 * len(y) / nclass))
    val_lb = int(round(0.2 * len(y)))

    indices = []
    for i in range(nclass):  # 每一个label
        h = tlx.convert_to_numpy((y == i))
        res = np.nonzero(h)
        res = np.array(res).reshape(-1)
        index = tlx.convert_to_tensor(res)
        n = tlx.get_tensor_shape(index)[0]
        index = index[
            np.random.permutation(n)
        ]
        indices.append(index)

    train_index = tlx.concat(values=[i[:percls_trn] for i in indices], axis=0)
    rest_index = tlx.concat(values=[i[percls_trn:] for i in indices], axis=0)
    m = tlx.get_tensor_shape(rest_index)[0]
    index2 = tlx.convert_to_tensor(np.random.permutation(m))
    index2 = tlx.cast(index2, dtype=tlx.int64)

    rest_index = rest_index[index2]
    valid_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    return train_index, valid_index, test_index

def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst
# utils：类
class DotProductAttention_tlx(tlx.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, querys, keys=None, values=None):
        """
        :param querys:  形状为[bsz=1,tgt_len==特征值repres数量==query数量，embedding的长度（特征值repre的长度）]
        :return:
        """
        # 默认自注意力
        keys = querys
        values = querys

        d = tlx.get_tensor_shape(querys)[-1]
        scores = tlx.ops.bmm(querys, tlx.nn.Transpose(perm=[0, 2, 1])(keys)) / math.sqrt(d)
        # 经验证两个类的scores是一样的
        self.attn_weights = tlx.nn.Softmax(axis=-1)(scores)
        return tlx.ops.bmm(self.attn_weights, values)
class MultiHeadAttention_tlx(tlx.nn.Module):
    def __init__(self, hidden_dim, nheads, tran_dropout=0.0):
        super().__init__(self)
        self.num_heads = nheads
        self.attention = DotProductAttention_tlx()

        self.W_q = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.W_k = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.W_v = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.W_o = tlx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, act=tlx.ReLU)

    def forward(self, q, k, v):
        is_batched = len(tlx.get_tensor_shape(q)) == 3

        if not is_batched:  # 如果输入没有batch
            q = tlx.ops.expand_dims(q, axis=0)
            k = tlx.ops.expand_dims(k, axis=0)
            v = tlx.ops.expand_dims(v, axis=0)

        q = transpose_qkv_tlx(self.W_q(q), self.num_heads)
        k = transpose_qkv_tlx(self.W_k(k), self.num_heads)
        v = transpose_qkv_tlx(self.W_v(v), self.num_heads)

        output = self.attention(q, k, v)
        output_concat = transepose_output_tlx(output, self.num_heads)
        res = self.W_o(output_concat)

        if not is_batched:  # 如果输入没有batch,之前加了batch，现在把batch消掉
            res = tlx.ops.squeeze(res, axis=0)

        return res
class SineEncoding(tlx.nn.Module):
    def __init__(self, hidden_dim=16):  # 128
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        initor = tlx.nn.initializers.XavierNormal()
        self.eig_w = tlx.nn.Linear(in_features=hidden_dim + 1, out_features=hidden_dim,  # h+1  -> h
                                   act=tlx.ReLU,
                                   W_init=initor)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]
        ee = e * self.constant
        div = tlx.ops.exp(tlx.ops.arange(0, self.hidden_dim, 2, dtype=tlx.float32) * (-math.log(10000) /
                                                                                      self.hidden_dim))
        pe = tlx.ops.expand_dims(ee, axis=1) * div
        eeig = tlx.ops.concat((tlx.ops.expand_dims(e, axis=1), tlx.ops.sin(pe), tlx.ops.cos(pe)),
                              axis=1)
        eeig = tlx.cast(eeig, dtype=tlx.float32)  # 改变类型
        # print(eeig)
        return self.eig_w(eeig)
class FeedForwardNetwork_tlx(tlx.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork_tlx, self).__init__()
        self.layer1 = tlx.nn.Linear(in_features=input_dim, out_features=hidden_dim, act=tlx.ReLU)
        self.layer2 = tlx.nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
class SpecLayer(tlx.nn.Module):  # 论文中将M+1映射到d的层
    #                        m        d
    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = tlx.nn.Dropout(p=prop_dropout)

        # self.weight 是一个[ 1 ,m ,d]
        if norm == 'none':
            self.weight = tlx.nn.Parameter(data=tlx.ones((1, nbases, ncombines)))
        else:  # 需要经过初始化
            self.weight = tlx.nn.Parameter(tlx.initializers.random_normal(mean=0.0, stddev=0.01) \
                                               (shape=(1, nbases, ncombines)))

        if norm == 'layer':  # Arxiv
            self.norm = tlx.nn.LayerNorm(ncombines)  # d
        elif norm == 'batch':  # Penn
            self.norm = tlx.nn.BatchNorm1d(num_features=ncombines)  # ncombines == d

        else:  # Others
            self.norm = None

    def forward(self, x):  # 输入是[N,m,d]，返回是[N,d]
        x = self.prop_dropout(x) * self.weight  # [N, m, d] * [1, m, d]  =  [ N , m, d ]，这里体现每一个节点特征dim
        x = tlx.ops.reduce_sum(x, axis=1)  # 返回[N , d]                                          都有一个filter

        if self.norm is not None:
            x = self.norm(x)
            x = tlx.nn.ReLU()(x)

        return x
class Specformer(tlx.nn.Module):

    def __init__(self, nclass, nfeat, nlayer=2, hidden_dim=32, nheads=4,
                 tran_dropout=0.2, feat_dropout=0.4, prop_dropout=0.5, norm='none'):
        super(Specformer, self).__init__()

        # 节点特征大小 nfeat ，特征值repre大小 hidden_dim
        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.feat_encoder = tlx.nn.Sequential(
            tlx.nn.Linear(in_features=nfeat, out_features=hidden_dim, act=tlx.ReLU),
            tlx.nn.Linear(in_features=hidden_dim, out_features=nclass),
        )
        self.linear_encoder = tlx.nn.Linear(in_features=nfeat, out_features=hidden_dim, act=tlx.ReLU)
        self.classify = tlx.nn.Linear(in_features=hidden_dim, out_features=nclass, act=tlx.ReLU)
        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = tlx.nn.Linear(in_features=hidden_dim, out_features=nheads, act=tlx.ReLU)
        self.mha_norm = tlx.nn.LayerNorm(hidden_dim)
        self.ffn_norm = tlx.nn.LayerNorm(hidden_dim)
        self.mha_dropout = tlx.nn.Dropout(p=tran_dropout)
        self.ffn_dropout = tlx.nn.Dropout(p=tran_dropout)
        self.mha = MultiHeadAttention_tlx(hidden_dim=hidden_dim, nheads=nheads, tran_dropout=tran_dropout)
        self.ffn = FeedForwardNetwork_tlx(hidden_dim, hidden_dim, hidden_dim)
        self.feat_dp1 = tlx.nn.Dropout(p=feat_dropout)
        self.feat_dp2 = tlx.nn.Dropout(p=feat_dropout)
        if norm == 'none':
            self.layers = tlx.nn.ModuleList(
                [SpecLayer(nheads + 1, nclass, prop_dropout, norm=norm) for i in range(nlayer)]
            )
        else:
            self.layers = tlx.nn.ModuleList(
                [SpecLayer(nheads + 1, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)]
            )

    def forward(self, x, edge_index):

        num_nodes = tlx.get_tensor_shape(x=x)[0]
        adj = np.zeros((num_nodes, num_nodes))
        source = edge_index[0]
        target = edge_index[1]
        for i in range(len(source)):
            adj[source[i], target[i]] = 1.
            adj[target[i], source[i]] = 1.

        # adj是图邻接矩阵，还不是L矩阵
        e, u = eigen_decompositon(adj)
        e = tlx.convert_to_tensor(e)
        u = tlx.convert_to_tensor(u)
        ut = tlx.nn.Transpose(perm=[1, 0])(u)

        x = tlx.cast(x=x, dtype=tlx.float32)
        ut = tlx.cast(x=ut, dtype=tlx.float32)
        e = tlx.cast(x=e, dtype=tlx.float32)
        u = tlx.cast(x=u, dtype=tlx.float32)

        # 对节点特征做变换
        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = tlx.cast(x=h, dtype=tlx.float32)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)  # nfeat->hidden_dim，把节点特征大小统一变换到特征值表示向量的大小

        # 对特征值repre做变换
        eig = self.eig_encoder(e)  # 输出eig为[N, d]

        mha_eig = self.mha_norm(eig)
        mha_eig = self.mha(q=mha_eig, k=mha_eig,v=mha_eig)
        eig = eig + self.mha_dropout(mha_eig)
        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)
        new_e = self.decoder(eig)  # 输出为[N, m]
        for conv in self.layers:
            basic_feats = [h]  # 一开始的h 是[N,d]的节点特征
            utx = ut @ h
            for i in range(self.nheads):
                basic_feats.append(u @ (tlx.ops.expand_dims(input=new_e[:, i], axis=1) * utx))  # [N, d]
            basic_feats = tlx.ops.stack(values=basic_feats, axis=1)  # [N, m, d]

            h = conv(basic_feats)  # conv层输入是[N,m,d]输出是[N,d]

        if self.norm == 'none':
            return h  # 返回[N,nlass]的logits
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h
class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'])

        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss



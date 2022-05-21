import torch as th
from torch.autograd import Function
#tensorized_layers:module_utils
import torch
from torch import nn as nn
from torch.nn import functional as F

def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    transform a batched graph to batched adjacency tensor and node feature tensor
    """
    batch_size = int(batch_adj.size()[0] / node_per_pool_graph) #20
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(batch_adj[start:end, start:end])
        feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: th.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: th.unsqueeze(x, 0), feat_list))
    adj = th.cat(adj_list, dim=0) #torch.cat() ；dim=0 20个list(1,12,12) --> (20,12,12)
    feat = th.cat(feat_list, dim=0)
    # adj:20 list (12 12) feat: 20 list (12,64)
    return feat, adj

#?
def masked_softmax(matrix, mask, dim=-1, memory_efficient=True,
                   mask_fill_value=-1e32):
    '''
    masked_softmax for dgl batch graph
    code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
    '''
    if mask is None:
        result = th.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = th.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = th.nn.functional.softmax(masked_matrix, dim=dim)
    return result

#return embedding
class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight, #tensor
            gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        num_node_per_graph = adj.size(1) #adj 20 12 12
        if self.use_bn and not hasattr(self, 'bn'):
            self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device) #nn.BatchNorm1d(dim)，dim等于前一层输出的维度。BatchNorm层输出的维度也是dim。
            #对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到比较标准的正态分布，
            # 使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(-1, keepdim=True)

        #adj 20 12 12; x 20 12 64
        h_k_N = torch.matmul(adj, x)  #20 12 64
        h_k = self.W(h_k_N) #linear 维度同上
        h_k = F.normalize(h_k, dim=2, p=2) #dim=2 对第三个维度（行）操作 【某一个维度除以那个维度对应的范数(默认是2范数)】
        h_k = F.relu(h_k)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k

    def __repr__(self):
        if self.use_bn:
            return 'BN' + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        # GNNl, pool(Al, Xl)
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1) #assignment matrix Sl = softmax(GNNl,pool(Al,Xl)) ;
        return s_l



class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.link_pred_layer = LinkPredLoss()
        #graphsage.py
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        #tensorized_layers.assignment
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())

    def forward(self, x, adj, log=False):
        # node feature matrix Zl= GNNl,embed(Al,Xl)    nl+1*d
        # embedding GNN为这一层的输入节点生成新的embedding
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log['s'] = s_l.cpu().numpy()

        # h:Xl+1 = SlT*Zl ; matmul():2个matrix的乘积
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        # Al+1 = SlT*Al*Sl    nl+1*nl+1
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log['a'] = anext.cpu().numpy()
        return xnext, anext

# import torch
# import torch.nn as nn
import tensorlayerx as tlx
import tensorlayerx.nn as nn
# import torch.nn.functional as F
import numpy as np


# import dgl.function as fn

# from examples.diffpool_dgl.model.dgl_layers.aggregator import MaxPoolAggregator, MeanAggregator, LSTMAggregator
# from examples.diffpool_dgl.model.dgl_layers.bundler import Bundler
# from gammagl.layers.conv import masked_softmax
# from examples.diffpool_dgl.model.loss import EntropyLoss


class GraphSageLayer(nn.Module):
    """
    GraphSage layer in Inductive learning paper by hamilton
    Here, graphsage layer is a reduced function in DGL framework
    """

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, bn=False, bias=True):
        super(GraphSageLayer, self).__init__()
        self.use_bn = bn
        self.bundler = Bundler(in_feats, out_feats, activation, dropout,
                               bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        if aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_feats, in_feats,
                                                activation, bias)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_feats, in_feats)
        else:
            self.aggregator = MeanAggregator()

    def forward(self, g, h):
        h = self.dropout(h)
        g.ndata['h'] = h
        if self.use_bn and not hasattr(self, 'bn'):
            device = h.device
            self.bn = nn.BatchNorm1d(h.size()[1]).to(device)
        g.update_all(tlx.nn.copy_src(src='h', out='m'), self.aggregator,
                     self.bundler)
        if self.use_bn:
            h = self.bn(h)
        h = g.ndata.pop('h')
        return h


class GraphSage(nn.Module):
    """
    Grahpsage network that concatenate several graphsage layer
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout, aggregator_type):
        super(GraphSage, self).__init__()
        self.layers = nn.LayerList([])

        # input layer
        self.layers.append(GraphSageLayer(in_feats, n_hidden, activation, dropout,
                                          aggregator_type))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GraphSageLayer(n_hidden, n_hidden, activation,
                                              dropout, aggregator_type))
        # output layer
        self.layers.append(GraphSageLayer(n_hidden, n_classes, None,
                                          dropout, aggregator_type))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


class DiffPoolBatchedGraphLayer(nn.Module):

    def __init__(self, input_dim, assign_dim, output_feat_dim,
                 activation, dropout, aggregator_type, link_pred):
        super(DiffPoolBatchedGraphLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = GraphSageLayer(
            input_dim,
            output_feat_dim,
            activation,
            dropout,
            aggregator_type)
        self.pool_gc = GraphSageLayer(
            input_dim,
            assign_dim,
            activation,
            dropout,
            aggregator_type)
        self.reg_loss = nn.LayerList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):#学习分配矩阵（g：Al h：Zl）
        #node feature matrix Zl= GNNl,embed(Al,Xl)    nl+1*d  embedding GNN为这一层的输入节点生成新的embedding
        feat = self.feat_gc(g, h)  # size = (sum_N, F_out), 775，64sum_N is num of nodes in this batch
        device = feat.device


        # assignment matrix Sl = softmax(GNNl,pool(Al,Xl)), pooling GNN生成从输入节点到nl+1个cluster的概率
        assign_tensor = self.pool_gc(g, h)  # size = (sum_N, N_a),775，12 N_a is num of nodes in pooled graph.
        assign_tensor = F.softmax(assign_tensor, dim=1) #assignment matrix Sl = softmax(GNNl,pool(Al,Xl)) ;
        assign_tensor = torch.split(assign_tensor, g.batch_num_nodes().tolist()) #20 -- batch size
        assign_tensor = torch.block_diag(*assign_tensor)  # size = (sum_N, batch_size * N_a) 775 240
                            #assign_tensor的diagonal矩阵

        #h:Xl+1 = SlT*Zl ; matmul():2个matrix的乘积
        h = torch.matmul(torch.t(assign_tensor), feat) #240，64
        adj = g.adjacency_matrix(transpose=True, ctx=device)#775，775

        #Al+1 = SlT*Al*Sl    nl+1*nl+1
        adj_new = torch.sparse.mm(adj, assign_tensor)#775，240
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)#240，240

        if self.link_pred:
            current_lp_loss = torch.norm(adj.to_dense() -
                                         torch.mm(assign_tensor, torch.t(assign_tensor))) / np.power(g.number_of_nodes(), 2)
            #torch.mm()矩阵乘法
            self.loss_log['LinkPredLoss'] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)
            #损失层 Loss 设置了一个损失函数用来比较网络的输出和目标值,通过最小化损失来驱动网络的训练。
            # 网络的损失通过前向操作计算,网络参数相对于损失函数的梯度则通过反向操作计算。

        return adj_new, h #240，240；240，64
               #Al+1，Xl+1




class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        #创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数，K是probs参数的长度。也就是说，按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
        #如果probs是长度为K的一维列表，则每个元素是对该索引处的类进行采样的相对概率。如果probs是二维的，它被视为一批概率向量
        assert not torch.isnan(entropy)
        return entropy

#链路预测
class LinkPredLoss(nn.Module):

    def forward(self, adj, anext, s_l):
        link_pred_loss = (
            adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()


class Aggregator(nn.Module):
    """
    Base Aggregator class. Adapting
    from PR# 403

    This class is not supposed to be called
    """

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        # N x F
        raise NotImplementedError


class MeanAggregator(Aggregator):
    '''
    Mean Aggregator for graphsage
    '''

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    '''
    Maxpooling aggregator for graphsage
    '''

    def __init__(self, in_feats, out_feats, activation, bias):
        super(MaxPoolAggregator, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation
        # Xavier initialization of weight
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain('relu'))

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    '''
    LSTM aggregator for graphsage
    '''

    def __init__(self, in_feats, hidden_feats):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        """
        Defaulted to initialite all zero
        """
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def aggre(self, neighbours):
        '''
        aggregation function
        '''
        # N X F
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0],
                                                            neighbours.size()[
            1],
            -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}


class Bundler(nn.Module):
    """
    Bundler, which will be the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(Bundler, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=in_feats * 2, out_features=out_feats, b_init=bias,act=activation)
        self.activation = activation
        self.linear.all_weights=tlx.initializers.XavierUniform
        # nn.init.xavier_uniform_(self.linear.weight,
        #                         gain=nn.init.calculate_gain('relu'))


    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}

import tensorlayerx as tlx
# from gammagl.layers.conv import xConv
import numpy as np
from gammagl.models.set2set import Set2Set

# GCN basic operation
class GraphConv(tlx.nn.Module):
    @property
    def all_weights(self):
        return self._all_weights

    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = tlx.nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim  # 89 #64
        self.output_dim = output_dim  # 64 #100
        initor = tlx.initializers.TruncatedNormal()
        self.weight = self._get_weights("weight", shape=(input_dim, output_dim),init=initor)#.cuda()  # 89,64 #64,100
                        #nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = self._get_weights("bias", shape=(output_dim,),init=initor)#.cuda() #nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = tlx.ops.matmul(adj, x)
        if self.add_self:
            y += x
        #print(y.is_cuda, self.weight.is_cuda) #true false
        #device = 'cuda'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')''
        #self.weight.to(device)
        #print(y.is_cuda, self.weight.is_cuda)
        y = tlx.ops.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = tlx.ops.l2_normalize(y, axis=2) #l2_nor [y = F.normalize(y, p=2, dim=2)]
            # print(y[0][0])
        return y

    @all_weights.setter
    def all_weights(self, value):
        self._all_weights = value


class GcnEncoderGraph(tlx.nn.Module):
    r"""
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.
    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters
    ----------
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        embedding_dim (int): embedding dimension
        label_dim (int): label dimension
        num_layers (int): number of layers
        pred_hidden_dims (int[]): predict hidden dimension
        concat (boolean): concat to get predict input dimension
        bn (boolean): batch normalization
        drop_rate (float): dropout rate
        args (parser): parser
    """

    @property
    def all_weights(self):
        return self._all_weights

    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = tlx.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        if isinstance(self, GraphConv):
            self.all_weights = tlx.initializers.XavierUniform(self.all_weights)
        if self.bias is not None:
            self.bias = tlx.initializers.constant(0.0)
        # for m in self.modules():#modules():
        #     if isinstance(m, GraphConv):
        #         m.weight.data = tlx.initializers.XavierUniform(m.weight.data)
        #                                          #(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         if m.bias is not None:
        #             m.bias.data = tlx.initializers.constant(0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)  # weight89,64; bias64
        conv_block = tlx.nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize,
                              bias=self.bias)  # output_dim=assign_dim=embedding_dim=100 #weight 64,100
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = tlx.nn.Linear(in_features=pred_input_dim, out_features=label_dim) #in&out
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(tlx.nn.Linear(in_features=pred_input_dim, out_features=pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(tlx.nn.Linear(pred_dim, label_dim))
            pred_model = tlx.nn.SequentialLayer(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks

        tup= []
        for num in batch_num_nodes:
            tup.append(int(num))
        packed_masks = tlx.ones(shape=(1,tup[0]))#!@@!
        # packed_masks = [tlx.ops.ones(int(num) for num in batch_num_nodes)]
        #packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = tlx.ops.zeros(shape=(batch_size, max_nodes))
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        #bn_module = tlx.nn.BatchNorm1d(x.size()[1]).cuda() #20，59，20 --》对59归一
        device = 'cpu'
        bn_module=tlx.nn.BatchNorm1d()(x.to(device)) #.size()[1]
        #decay momentum 0.1
        return bn_module #(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = tlx.ops.concat(x_all, 2)#x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        #mask
        #max_num_nodes = np.size(adj)
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj) #索引矩阵，索引矩阵的行，索引矩阵的列。
        x = self.act(x)
        # print(x.is_cuda)
        if self.bn:
            x = self.apply_bn(x) #59nawei
        out_all = []
        out = tlx.ops.reduce_max(x, axis=1) #out, _ = torch.max(x, dim=1) 每行的最大值
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out = tlx.ops.reduce_max(x, 1) #out, _每行的最大值
            out_all.append(out)
            if self.num_aggs == 2:
                out = tlx.ops.cumsum(x, 1) #torch.sum(x,dim=1) 每行的和
                out_all.append(out)
        x = self.conv_last(x, adj)
        # x = self.act(x)
        out= tlx.ops.reduce_max(x, 1) #out, _
        out_all.append(out)
        if self.num_aggs == 2:
            out = tlx.ops.cumsum(x, 1)
            out_all.append(out)
        if self.concat:
            output = tlx.ops.concat(out_all, 1) #沿行拼接 dim=1
        else:
            output = out #20,60
        ypred = self.pred_model(output) #20,2
        # print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return tlx.losses.softmax_cross_entropy_with_logits(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = tlx.ops.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return tlx.losses.binary_cross_entropy(pred, label_onehot) #!!!!!!
            #return torch.nn.MultiLabelMarginLoss()(pred, label_onehot) #multi-label one-versus-all 多分类
        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())

    @all_weights.setter
    def all_weights(self, value):
        self._all_weights = value


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        # out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        self.conv_first_after_pool = tlx.nn.ModuleList()
        self.conv_block_after_pool = tlx.nn.ModuleList()
        self.conv_last_after_pool = tlx.nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = tlx.nn.ModuleList()
        self.assign_conv_block_modules = tlx.nn.ModuleList()
        self.assign_conv_last_modules = tlx.nn.ModuleList()
        self.assign_pred_modules = tlx.nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)  # 100
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim  # 228
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.named_children(): #modules()
            if isinstance(m, GraphConv):
                m.weight.data = tlx.initializers.XavierUniform(m.weight.data)# gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = tlx.initializers.constant(0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = tlx.ops.reduce_max(embedding_tensor, 1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = tlx.ops.cumsum(embedding_tensor, 1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = tlx.nn.Softmax()(self.assign_pred_modules[i](self.assign_tensor))
            #nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = tlx.ops.matmul(tlx.ops.transpose(self.assign_tensor), embedding_tensor)
                                #torch.transpose(self.assign_tensor, 1, 2), embedding_tensor
            adj = tlx.ops.transpose(self.assign_tensor) @ adj @ self.assign_tensor
                #torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = tlx.ops.reduce_max(embedding_tensor, 1)
            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = tlx.ops.cumsum(embedding_tensor, 1)
                out_all.append(out)

        if self.concat:
            output = tlx.ops.concat(out_all, 1) #(out_all) ?
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ tlx.ops.transpose(self.assign_tensor)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = tlx.ops.reduce_min(pred_adj, tlx.ops.ones(1, dtype=pred_adj.dtype).cuda())
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * tlx.ops.log(pred_adj + eps) - (1 - adj) * tlx.ops.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ tlx.ops.transpose(embedding_mask)
                self.link_loss[(1 - adj_mask).bool()] = 0.0

            self.link_loss = tlx.ops.cumsum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss




import tensorlayerx as tlx
import numpy as np

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
        self.input_dim = input_dim
        self.output_dim = output_dim
        initor = tlx.initializers.TruncatedNormal()
        self.weight = self._get_weights("weight", shape=(input_dim, output_dim),init=initor)
        if bias:
            self.bias = self._get_weights("bias", shape=(output_dim,),init=initor)
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = tlx.ops.matmul(adj, x)
        if self.add_self:
            y += x
        y = tlx.ops.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = tlx.ops.l2_normalize(y, axis=2)
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

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = tlx.nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize,
                              bias=self.bias)
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
        packed_masks = tlx.ones(shape=(1,tup[0]))
        batch_size = len(batch_num_nodes)
        out_tensor = tlx.ops.zeros(shape=(batch_size, max_nodes))
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        device = 'cpu'
        bn_module=tlx.nn.BatchNorm1d()(x.to(device))
        return bn_module

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
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = tlx.ops.concat(x_all, 2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        # print(x.is_cuda)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out = tlx.ops.reduce_max(x, axis=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out = tlx.ops.reduce_max(x, 1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = tlx.ops.cumsum(x, 1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        out= tlx.ops.reduce_max(x, 1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = tlx.ops.cumsum(x, 1)
            out_all.append(out)
        if self.concat:
            output = tlx.ops.concat(out_all, 1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return tlx.losses.softmax_cross_entropy_with_logits(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            diag = tlx.convert_to_tensor(np.eye(tlx.reduce_max(label) + 1), dtype=tlx.int64)
            label_onehot=tlx.gather(diag, label)
            return tlx.losses.binary_cross_entropy(pred, label_onehot) #MultiLabelMarginLoss()(pred, label_onehot)

    @all_weights.setter
    def all_weights(self, value):
        self._all_weights = value






import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing



def segment_softmax(weight, segment_ids, num_nodes):
    #减去最大值，避免exp()后溢出
    max_values = tlx.ops.unsorted_segment_max(weight, segment_ids, num_segments = num_nodes)
    gathered_max_values = tlx.ops.gather(max_values, segment_ids)
    weight = weight - gathered_max_values
    exp_weight = tlx.ops.exp(weight)

    sum_weights = tlx.ops.unsorted_segment_sum(exp_weight, segment_ids, num_segments = num_nodes)
    sum_weights = tlx.ops.gather(sum_weights, segment_ids)
    softmax_weight = tlx.ops.divide(exp_weight, sum_weights)

    return softmax_weight


class AGNNConv(MessagePassing):
    r'''The graph attention operator from the `"Attention-based Graph Neural Network for Semi-supervised Learning"
    <http://arxiv.org/abs/1803.03735>`_paper

    .. math::





    Args:
        in_channels(int or tuple)
    '''


    
    def __init__(self,
                in_channels,
                out_channels,
                edge_index,
                num_nodes,
                require_grad = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        if(require_grad == True):
            initor = tlx.initializers.RandomUniform(0,1)
            self.beta = self._get_weights("beta", shape = [1], init = initor)
        else:
            initor = tlx.initializers.Zeros()
            self.beta = self._get_weights("beta", shape = [1], init = initor, trainable = False)

        
    def message(self, x, edge_index, edge_weight = None, num_nodes = None):
        node_src = edge_index[0, :]
        node_dst = edge_index[1, :]

        x_src = tlx.ops.gather(x, node_src)
        x_dst = tlx.ops.gather(x, node_dst)
        
        cos = tlx.ops.reduce_sum(
            tlx.ops.l2_normalize(x_src) * tlx.ops.l2_normalize(x_dst), axis = -1, keepdims=True)
        unsoftmax_weight = cos * self.beta

        softmax_weight = segment_softmax(unsoftmax_weight, node_dst, num_nodes)
    
        return  softmax_weight * x_src  


    def forward(self, x):
        return self.propagate(x, self.edge_index, num_nodes = self.num_nodes)
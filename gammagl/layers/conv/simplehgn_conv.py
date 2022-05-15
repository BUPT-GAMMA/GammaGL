import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

class SimpleHGN(MessagePassing):
    def __init__(self,
                in_src_feats,
                in_dst_feats,
                out_feats,
                edge_types,
                edge_feats,
                heads=1,
                concat=True,
                negetive_slop=0.2,
                feat_drop=0.,
                attn_drop=0.,
                residual=False,
                activation=None,
                bias=None,
                beta=0.,
                allow_zero_in_degree=False):
        super().__init__()

        self.in_src_feats = in_src_feats
        self.in_dst_feats = in_dst_feats
        self.out_channels = out_feats
        self.edge_types = edge_types
        self.edge_feats = edge_feats
        self.heads = heads
        self.allow_zero_in_degree = allow_zero_in_degree

        self.edge_embedding = tlx.nn.Embedding(edge_types, edge_feats)
        #TODO:初始化参数需要的gain，tlx框架下没有提供
        self.fc_src = tlx.nn.Linear(out_feats * heads, b_init=None)
        self.fc_dst = tlx.nn.Linear(out_feats * heads, b_init=None)
        self.fc_e = tlx.nn.Linear(edge_feats * eads, b_init=None)

        self.attn_l = self._get_weights('attn_l', shape=(1, heads, out_feats))
        self.attn_r = self._get_weights('attn_r', shape=(1, heads, out_feats))
        self.attn_e = self._get_weights('attn_r', shape=(1, heads, edge_feats))

        self.feat_drop = tlx.nn.Dropout(feat_drop)
        self.attn_drop = tlx.nn.Dropout(attn_drop)
        self.leaky_relu = tlx.nn.LeakyReLU(negative_slope)

        if residual:
            if(self.in_dst_feats != out_feats):
                self.fc_res = tlx.nn.Linear(num_heads * out_feats, b_init=None)
            else:
                #self.fc_res = 
                TODO()
        self.activation = activation
        self.bias = bias


        

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from layers.conv.gcn_forheco import metapathSpecificGCN
from layers.attention.meta_path_attention import Attention
from layers.attention.network_schema_attention import intra_att
from layers.attention.network_schema_attention import inter_att
import os
class Sc_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop):
        super(Sc_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num

    def forward(self, nei_h, nei_index):
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    if os.environ['TL_BACKEND'] == 'mindspore':
                        per_node_nei_tmp = []
                        for k in range (len(per_node_nei)):
                            per_node_nei_tmp.append(per_node_nei[k].numpy())
                        select_one = tlx.convert_to_tensor(np.random.choice(per_node_nei_tmp, sample_num,
                                                               replace=False))
                    else:
                        select_one = tlx.convert_to_tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    if os.environ['TL_BACKEND'] == 'mindspore':
                        per_node_nei_tmp = []
                        for k in range (len(per_node_nei)):
                            per_node_nei_tmp.append(per_node_nei[k].numpy())
                        select_one = tlx.convert_to_tensor(np.random.choice(per_node_nei_tmp, sample_num,
                                                               replace=True))
                    else:
                        select_one = tlx.convert_to_tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = tlx.concat(sele_nei, axis=0)
            one_type_emb = tlx.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0])) 
            embeds.append(one_type_emb)
        z_mc = self.inter(embeds)
        return z_mc
class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([metapathSpecificGCN(hidden_dim, hidden_dim) for _ in range(P)]) 
        self.att = Attention(hidden_dim, attn_drop)

    def forward(self, h, mps):
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
        z_mp = self.att(embeds)
        return z_mp
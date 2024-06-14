import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn


class metapathSpecificGCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(metapathSpecificGCN, self).__init__()
        self.fc = nn.Linear(in_features=in_ft, out_features=out_ft, W_init="he_normal")
        self.act = nn.LeakyReLU()

        if bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, out_ft), init=initor)
        else:
            self.register_parameter('bias', None)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = tlx.matmul(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='xavier_normal')

        self.tanh = nn.Tanh()
        initor = tlx.initializers.XavierNormal(gain=1.414)
        self.att = self._get_weights("att", shape=(1, hidden_dim), init=initor)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        cnt = 0
        for embed in embeds:
            cnt = cnt + 1
            sp = tlx.reduce_mean(self.tanh(self.fc(embed)), axis=0)
            if cnt == 1:
                attn_curr_array = tlx.convert_to_numpy(attn_curr)
            sp = tlx.transpose(sp)
            sp = tlx.convert_to_numpy(sp)
            beta_tmp = np.matmul(attn_curr_array[0], sp)
            beta_tmp = tlx.expand_dims(tlx.convert_to_tensor(beta_tmp), 0)
            beta.append(beta_tmp)

        beta = tlx.reshape(tlx.concat(beta, axis=-1), (-1,))
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        initor = tlx.initializers.XavierNormal(gain=1.414)
        self.att = self._get_weights("att", shape=(1, 2 * hidden_dim), init=initor)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(axis=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = []
        length = tlx.get_tensor_shape(nei)[0]
        for i in range(length):
            temp = tlx.gather(h, nei[i])
            nei_emb.append(tlx.convert_to_numpy(temp))
        nei_emb = tlx.convert_to_tensor(nei_emb)
        h_refer = tlx.expand_dims(h_refer, 1)
        h_refer = tlx.concat([h_refer] * tlx.get_tensor_shape(nei_emb)[1], axis=1)
        all_emb = tlx.concat([h_refer, nei_emb], axis=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(tlx.matmul(all_emb, tlx.transpose(attn_curr)))
        att = self.softmax(att)
        nei_emb = tlx.reduce_sum(att * nei_emb, axis=1)
        return nei_emb


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='xavier_normal')
        self.tanh = nn.Tanh()

        initor = tlx.initializers.XavierNormal(gain=1.414)
        self.att = self._get_weights("att", shape=(1, hidden_dim), init=initor)
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        cnt = 0
        for embed in embeds:
            cnt = cnt + 1
            sp = tlx.reduce_mean(self.tanh(self.fc(embed)), axis=0)
            if cnt == 1:
                attn_curr_array = tlx.convert_to_numpy(attn_curr)
            sp = tlx.transpose(sp)
            sp = tlx.convert_to_numpy(sp)
            beta_tmp = np.matmul(attn_curr_array[0], sp)
            beta_tmp = tlx.expand_dims(tlx.convert_to_tensor(beta_tmp), 0)
            beta.append(beta_tmp)

        beta = tlx.reshape(tlx.concat(beta, axis=-1), (-1,))
        beta = self.softmax(beta)
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp


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
                    select_one = tlx.convert_to_tensor(np.random.choice(per_node_nei, sample_num,
                                                                        replace=False))[np.newaxis]
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

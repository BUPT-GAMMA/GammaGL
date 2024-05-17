import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import os


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
            if cnt == 1 :
                attn_curr_array = tlx.convert_to_numpy(attn_curr)
            sp = tlx.transpose(sp)
            sp = tlx.convert_to_numpy(sp)
            beta_tmp = np.matmul(attn_curr_array[0], sp)
            beta_tmp = tlx.expand_dims(tlx.convert_to_tensor(beta_tmp), 0)
            beta.append(beta_tmp)
            
        beta = tlx.reshape(tlx.concat(beta, axis=-1), (-1, ))  
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        initor = tlx.initializers.XavierNormal(gain=1.414)
        self.att = self._get_weights("att", shape=(1, 2*hidden_dim), init=initor)
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
        nei_emb = tlx.reduce_sum(att*nei_emb, axis=1)
        return nei_emb 
import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import os
class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='xavier_normal') 
        # self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()

        initor = tlx.initializers.XavierNormal(gain=1.414)
        self.att = self._get_weights("att", shape=(1, hidden_dim), init=initor)
        # self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        # nn.init.xavier_normal_(self.att.data, gain=1.414)

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
            #print(attn_curr_array[0])
            #print(tlx.matmul(attn_curr, tlx.transpose(sp)))
            if os.environ['TL_BACKEND'] == 'mindspore':
                sp = np.array(sp)
                sp = np.transpose(sp)
                #print(sp)
                #print(attn_curr_array[0])
                #sp_expand = np.expand_dims(sp, 0)
                beta_tmp = np.matmul(attn_curr_array[0], sp)
            else:
                sp = tlx.transpose(sp)
                sp = tlx.convert_to_numpy(sp)
                beta_tmp = np.matmul(attn_curr_array[0], sp)
            beta_tmp = tlx.expand_dims(tlx.convert_to_tensor(beta_tmp), 0)
            beta.append(beta_tmp)
        
        beta = tlx.reshape(tlx.concat(beta, axis=-1), (-1, ))
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp

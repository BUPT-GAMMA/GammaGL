import tensorlayerx as tlx
from tensorlayerx import nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(
            in_features=n_h, 
            out_features=n_h, 
            W_init=tlx.initializers.XavierUniform(),
            b_init=tlx.initializers.Zeros()
        )
        self.proj = None
        self.c_dim = None
        self.h_dim = None

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_dim = tlx.get_tensor_shape(c)[-1]
        h_dim = tlx.get_tensor_shape(h_pl)[-1]
        batch_size = tlx.get_tensor_shape(h_pl)[0]
        
        if c_dim != h_dim and (self.proj is None or c_dim != self.c_dim or h_dim != self.h_dim):
            self.c_dim = c_dim
            self.h_dim = h_dim
            self.proj = nn.Linear(
                in_features=c_dim, 
                out_features=h_dim, 
                W_init=tlx.initializers.XavierUniform(),
                b_init=tlx.initializers.Zeros()
            )
        
        if c_dim != h_dim:
            c_proj = self.proj(c)
        else:
            c_proj = c
        
        c_x = tlx.expand_dims(c_proj, 0)
        c_x = tlx.tile(c_x, [batch_size, 1])

        h_pl_proj = self.fc1(h_pl)  

        sc_1 = tlx.reduce_sum(h_pl_proj * c_x, axis=1) 
        
        h_mi_proj = self.fc1(h_mi)  

        sc_2 = tlx.reduce_sum(h_mi_proj * c_x, axis=1)  
        
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        
        logits = tlx.concat([sc_1, sc_2], axis=0)
        
        return logits

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='he_normal'),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, W_init='he_normal')
        )
        self.tau = tau
        self.lam = lam
    def sim(self, z1, z2):
        z1_norm = tlx.l2_normalize(z1, axis=-1)
        z2_norm = tlx.l2_normalize(z2, axis=-1)
        z1_norm = tlx.reshape(tlx.reduce_mean(z1/z1_norm, axis=-1), (-1, 1))
        z2_norm = tlx.reshape(tlx.reduce_mean(z2/z2_norm, axis=-1), (-1, 1))
        dot_numerator = tlx.matmul(z1, tlx.transpose(z2))
        dot_denominator = tlx.matmul(z1_norm, tlx.transpose(z2_norm))
        sim_matrix = tlx.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self , z, pos):
        z_mp = z.get("z_mp")
        z_sc = z.get("z_sc")
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = tlx.transpose(matrix_mp2sc)
        
        matrix_mp2sc = matrix_mp2sc / (tlx.reshape(tlx.reduce_sum(matrix_mp2sc, axis=1), (-1, 1)) + 1e-8)
        lori_mp = -tlx.reduce_mean(tlx.log(tlx.reduce_sum(tlx.multiply(matrix_mp2sc, pos), axis=-1)))

        matrix_sc2mp = matrix_sc2mp / (tlx.reshape(tlx.reduce_sum(matrix_sc2mp, axis=1), (-1, 1)) + 1e-8)
        lori_sc = -tlx.reduce_mean(tlx.log(tlx.reduce_sum(tlx.multiply(matrix_sc2mp, pos), axis=-1)))
        return self.lam * lori_mp + (1 - self.lam) * lori_sc

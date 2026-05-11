import os
os.environ.setdefault('TL_BACKEND', 'torch')

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import torch
from gammagl.layers.conv.rgt_layers import EuclideanEncoder, ManifoldEncoder
from gammagl.layers.attention.rgt_attention import HyperbolicStructureLearner, SphericalStructureLearner, EuclideanStructureLearner
from gammagl.layers.manifolds import Lorentz, Sphere, ProductSpace, Euclidean
from gammagl.layers.vector_quantize.vq_euclidean import VectorQuantize_E
from gammagl.layers.vector_quantize.vq_riemann import VectorQuantize_R


def _tensor_has_nan(x):
    if isinstance(x, torch.Tensor):
        return bool(torch.isnan(x).any().item())
    return bool(np.isnan(tlx.convert_to_numpy(x)).any())


def _sanitize_tensor(x, clip=1e4):
    if isinstance(x, torch.Tensor):
        return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    return tlx.clip_by_value(tlx.where(tlx.isnan(x) | tlx.isinf(x),
                                        tlx.zeros_like(x), x),
                             -clip, clip)


class InitBlock(tlx.nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, bias, activation, dropout):
        super().__init__()
        self.Euc_init = EuclideanEncoder(in_dim, hidden_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, in_dim, hidden_dim, out_dim, bias, None, 0.)
        self.Sph_init = ManifoldEncoder(manifold_S, in_dim, hidden_dim, out_dim, bias, None, 0.)

    def forward(self, edge_index, tokens):
        E = self.Euc_init(tokens)
        H = self.Hyp_init(tokens, edge_index)
        S = self.Sph_init(tokens, edge_index)
        return E, H, S


class StructuralBlock(tlx.nn.Module):
    def __init__(self, manifold_H, manifold_S, manifold_E, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.manifold_E = manifold_E
        self.Hyp_learner = HyperbolicStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.Sph_learner = SphericalStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.Euc_learner = EuclideanStructureLearner(self.manifold_E, in_dim, hidden_dim, out_dim, dropout)

        self.proj = nn.Sequential(
            tlx.layers.Linear(in_features=3 * out_dim, out_features=hidden_dim),
            tlx.nn.ReLU(),
            tlx.layers.Linear(in_features=hidden_dim, out_features=out_dim)
        )

    def forward(self, x_tuple, data):
        x_E, x_H, x_S = x_tuple
        x_H = self.Hyp_learner(x_H, x_S, data.batch_tree)
        x_S = self.Sph_learner(x_H, x_S, data.batch_cycle)
        x_E = self.Euc_learner(x_E, data.batch_sequence)

        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_S.proju(x_S, x_E)

        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp0back(x_S, S_E)

        E = tlx.concat([x_E, H_E, S_E], axis=-1)
        x_E = self.proj(E)
        x_E = x_E / (tlx.sqrt(tlx.reduce_sum(x_E * x_E, axis=-1, keepdims=True)) + 1e-8)
        return x_E, x_H, x_S


class VQBlock(tlx.nn.Module):
    def __init__(self, args, manifold_H, manifold_S, manifold_E):
        super().__init__()
        n_layers = getattr(args, 'n_layers', 3)
        vq_dim = getattr(args, 'hidden_dim_vq', getattr(args, 'd_hidden', getattr(args, 'hidden_dim', 32)))
        code_dim = getattr(args, 'code_dim', getattr(args, 'd_hidden', getattr(args, 'hidden_dim', 32)))
        codebook_head = getattr(args, 'codebook_head', 8)
        codebook_decay = getattr(args, 'codebook_decay', 0.8)
        commit_weight = getattr(args, 'commit_weight', 0.25)
        ortho_reg_weight = getattr(args, 'ortho_reg_weight', 0.0)
        ortho_reg_max_codes = getattr(args, 'ortho_reg_max_codes', 256)

        self.Euc_vq = VectorQuantize_E(
            manifold=manifold_E,
            dim=vq_dim,
            codebook_size=args.codebook_size,
            codebook_dim=code_dim,
            heads=codebook_head,
            separate_codebook_per_head=True,
            decay=codebook_decay,
            commitment_weight=commit_weight,
            use_cosine_sim=True,
            orthogonal_reg_weight=ortho_reg_weight,
            orthogonal_reg_max_codes=ortho_reg_max_codes,
            orthogonal_reg_active_codes_only=False,
            kmeans_init=False,
            ema_update=False,
            sync_codebook=False,
            learnable_codebook=True,
            sample_codebook_temp=1.,
            threshold_ema_dead_code=2,
        )
        self.Hyp_vq = VectorQuantize_R(
            manifold=manifold_H,
            dim=vq_dim,
            codebook_size=args.codebook_size,
            codebook_dim=code_dim,
            heads=codebook_head,
            separate_codebook_per_head=True,
            decay=codebook_decay,
            commitment_weight=commit_weight,
            use_cosine_sim=True,
            orthogonal_reg_weight=ortho_reg_weight,
            orthogonal_reg_max_codes=ortho_reg_max_codes,
            orthogonal_reg_active_codes_only=False,
            kmeans_init=False,
            ema_update=False,
            learnable_codebook=True,
            sync_codebook=False,
            sample_codebook_temp=1.,
            threshold_ema_dead_code=2,
        )
        self.Sph_vq = VectorQuantize_R(
            manifold=manifold_S,
            dim=vq_dim,
            codebook_size=args.codebook_size,
            codebook_dim=code_dim,
            heads=codebook_head,
            separate_codebook_per_head=True,
            decay=codebook_decay,
            commitment_weight=commit_weight,
            use_cosine_sim=True,
            orthogonal_reg_weight=ortho_reg_weight,
            orthogonal_reg_max_codes=ortho_reg_max_codes,
            orthogonal_reg_active_codes_only=False,
            kmeans_init=False,
            ema_update=False,
            learnable_codebook=True,
            sync_codebook=False,
            sample_codebook_temp=1.,
            threshold_ema_dead_code=2,
        )

    def forward(self, x_E, x_H, x_S):
        quantize_E, indices_E, commit_loss_E, _ = self.Euc_vq(x_E)
        quantize_H, indices_H, commit_loss_H, _ = self.Hyp_vq(x_H)
        quantize_S, indices_S, commit_loss_S, _ = self.Sph_vq(x_S)
        if isinstance(quantize_E, torch.Tensor) and torch.isnan(quantize_E).any():
            quantize_E = x_E
            commit_loss_E = tlx.zeros_like(commit_loss_E)
        if isinstance(quantize_H, torch.Tensor) and torch.isnan(quantize_H).any():
            quantize_H = x_H
            commit_loss_H = tlx.zeros_like(commit_loss_H)
        if isinstance(quantize_S, torch.Tensor) and torch.isnan(quantize_S).any():
            quantize_S = x_S
            commit_loss_S = tlx.zeros_like(commit_loss_S)
        return quantize_E, quantize_H, quantize_S, indices_E, indices_H, indices_S, commit_loss_E, commit_loss_H, commit_loss_S


class RGT(tlx.nn.Module):
    def __init__(self, args=None, **kwargs):
        super(RGT, self).__init__()

        if args is not None and not isinstance(args, dict) and not kwargs:
            n_layers = getattr(args, 'n_layers', 3)
            in_dim = args.num_features
            hidden_dim = args.d_hidden
            embed_dim = getattr(args, 'embed_dim', args.d_hidden)
            bias = getattr(args, 'bias', True)
            activation = getattr(args, 'activation', tlx.relu)
            dropout = getattr(args, 'dropout', 0.1)
        else:
            configs = kwargs.get('configs', None)
            n_layers = kwargs.get('n_layers', getattr(configs, 'n_layers', 3) if configs else 3)
            in_dim = kwargs.get('in_dim', getattr(configs, 'num_features', 32) if configs else 32)
            hidden_dim = kwargs.get('hidden_dim', getattr(configs, 'd_hidden', 256) if configs else 256)
            embed_dim = kwargs.get('embed_dim', getattr(configs, 'embed_dim', 32) if configs else 32)
            bias = kwargs.get('bias', getattr(configs, 'bias', True) if configs else True)
            activation = kwargs.get('activation', getattr(configs, 'activation', tlx.relu) if configs else tlx.relu)
            dropout = kwargs.get('dropout', getattr(configs, 'dropout', 0.1) if configs else 0.1)
            if configs is not None:
                args = configs

        self.manifold_H = Lorentz()
        self.manifold_S = Sphere()
        self.manifold_E = Euclidean()
        self.product = ProductSpace(*[(self.manifold_H, embed_dim),
                                      (self.manifold_S, embed_dim)])
        self.init_block = InitBlock(self.manifold_H, self.manifold_S,
                                    embed_dim, hidden_dim, embed_dim, bias,
                                    activation, dropout)
        self.blocks = nn.ModuleList([
            StructuralBlock(self.manifold_H, self.manifold_S, self.manifold_E,
                            embed_dim, hidden_dim, embed_dim, dropout)
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            tlx.layers.Linear(in_features=2 * embed_dim, out_features=hidden_dim),
            tlx.nn.ReLU(),
            tlx.layers.Linear(in_features=hidden_dim, out_features=embed_dim)
        )
        self.vqblock = VQBlock(args, self.manifold_H, self.manifold_S, self.manifold_E)

    @staticmethod
    def _check_nan(tag, *tensors):
        for i, t in enumerate(tensors):
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any():
                    print(f"[NaN TRACE] {tag}[{i}]: NaN in tensor shape={t.shape}, nan_count={torch.isnan(t).sum().item()}")
                    return True
                if torch.isinf(t).any():
                    print(f"[NaN TRACE] {tag}[{i}]: Inf in tensor shape={t.shape}")
                    return True
            elif isinstance(t, np.ndarray):
                if np.isnan(t).any():
                    print(f"[NaN TRACE] {tag}[{i}]: NaN in ndarray shape={t.shape}")
                    return True
            elif hasattr(t, 'numpy'):
                arr = tlx.convert_to_numpy(t)
                if np.isnan(arr).any():
                    print(f"[NaN TRACE] {tag}[{i}]: NaN in TLX tensor shape={arr.shape}")
                    return True
        return False

    def forward(self, data):
        n_id = data.n_id if hasattr(data, 'n_id') and data.n_id is not None else None

        tokens = data.tokens(n_id) if callable(data.tokens) else data.tokens
        if not isinstance(tokens, torch.Tensor):
            tokens = tlx.convert_to_tensor(tokens, dtype=tlx.float32)
        else:
            tokens = tokens.to(dtype=torch.float32)
        tokens = _sanitize_tensor(tokens)

        edge_index = data.edge_index
        if isinstance(edge_index, torch.Tensor):
            data.edge_index = edge_index.to(dtype=torch.int64)
        else:
            data.edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)

        x_E, x_H, x_S = self.init_block(data.edge_index, tokens)
        x_E = _sanitize_tensor(x_E)
        x_H = _sanitize_tensor(x_H)
        x_S = _sanitize_tensor(x_S)

        for i, block in enumerate(self.blocks):
            x_E, x_H, x_S = block((x_E, x_H, x_S), data)
            x_E = _sanitize_tensor(x_E)
            x_H = _sanitize_tensor(x_H)
            x_S = _sanitize_tensor(x_S)

        q_E, q_H, q_S, indices_E, indices_H, indices_S, commit_loss_E, commit_loss_H, commit_loss_S = \
            self.vqblock(x_E, x_H, x_S)

        return x_E, x_H, x_S, q_E, q_H, q_S, commit_loss_E, commit_loss_H, commit_loss_S

    def loss(self, x_tuple):
        x_E, x_H, x_S, q_E, q_H, q_S, commit_loss_E, commit_loss_H, commit_loss_S = x_tuple

        loss_E = commit_loss_E + commit_loss_H + commit_loss_S

        H_E = self.manifold_H.proju(q_H, q_E)
        S_E = self.manifold_S.proju(q_S, q_E)

        H_E = self.manifold_H.transp0back(q_H, H_E)
        S_E = self.manifold_S.transp0back(q_S, S_E)
        E = tlx.reduce_mean(tlx.stack([H_E, S_E], axis=0), axis=0)

        log0_H = self.manifold_H.logmap0(q_H)
        log0_S = self.manifold_S.logmap0(q_S)
        H_E = self.proj(tlx.concat([log0_H, H_E], axis=-1))
        S_E = self.proj(tlx.concat([log0_S, S_E], axis=-1))
        loss_HS = self.cal_cl_loss(H_E, S_E)
        loss_HE = self.cal_cl_loss(H_E, E)
        loss_SE = self.cal_cl_loss(S_E, E)
        E = tlx.concat([E, H_E, S_E], axis=-1)

        loss = loss_E + 0.1 * loss_HS + 0.1 * loss_HE + 0.1 * loss_SE

        return loss, E

    def cal_cl_loss(self, x1, x2):
        EPS = 1e-6
        norm1 = tlx.sqrt(tlx.reduce_sum(x1 * x1, axis=-1))
        norm2 = tlx.sqrt(tlx.reduce_sum(x2 * x2, axis=-1))
        sim_matrix = tlx.matmul(x1, tlx.transpose(x2)) / (tlx.matmul(tlx.expand_dims(norm1, -1), tlx.expand_dims(norm2, -1).T) + EPS)
        sim_matrix = tlx.exp(sim_matrix / 0.2)
        pos_sim = tlx.diag(sim_matrix)
        loss_1 = pos_sim / (tlx.reduce_sum(sim_matrix, axis=-2) + EPS)
        loss_2 = pos_sim / (tlx.reduce_sum(sim_matrix, axis=-1) + EPS)

        loss_1 = -tlx.reduce_mean(tlx.log(loss_1))
        loss_2 = -tlx.reduce_mean(tlx.log(loss_2))
        loss = (loss_1 + loss_2) / 2.
        return loss

    def get_encoder(self, batch):
        n_id = batch.n_id if hasattr(batch, 'n_id') and batch.n_id is not None else None
        tokens = batch.tokens(n_id) if callable(batch.tokens) else batch.tokens
        if not isinstance(tokens, torch.Tensor):
            tokens = tlx.convert_to_tensor(tokens, dtype=tlx.float32)
        else:
            tokens = tokens.to(dtype=torch.float32)
        tokens = _sanitize_tensor(tokens)

        edge_index = batch.edge_index
        if isinstance(edge_index, torch.Tensor):
            batch.edge_index = edge_index.to(dtype=torch.int64)
        else:
            batch.edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64)

        x_E, x_H, x_S = self.init_block(batch.edge_index, tokens)
        x_E = _sanitize_tensor(x_E)
        x_H = _sanitize_tensor(x_H)
        x_S = _sanitize_tensor(x_S)

        for block in self.blocks:
            x_E, x_H, x_S = block((x_E, x_H, x_S), batch)
            x_E = _sanitize_tensor(x_E)
            x_H = _sanitize_tensor(x_H)
            x_S = _sanitize_tensor(x_S)

        q_E, q_H, q_S, indices_E, indices_H, indices_S, _, _, _ = \
            self.vqblock(x_E, x_H, x_S)

        log0_H = self.manifold_H.logmap0(q_H)
        log0_S = self.manifold_S.logmap0(q_S)

        return q_E, log0_H, log0_S

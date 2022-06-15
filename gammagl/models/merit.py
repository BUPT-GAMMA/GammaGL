from unicodedata import name
import tensorlayerx as tlx
import copy
import numpy as np
from gammagl.layers.conv import GCNConv


class MLP(tlx.nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        # self.net_list = []
        # self.net_list.append(tlx.nn.Linear(in_features=inp_size, out_features=hidden_size))
        # self.net_list.append(tlx.nn.BatchNorm1d(num_features=hidden_size))
        # self.net_list.append(tlx.nn.PRelu(hidden_size))
        # self.net_list.append(tlx.nn.Linear(in_features=hidden_size, out_features=outp_size))
        # self.net = tlx.nn.Sequential(self.net_list)
        self.net = tlx.nn.Sequential([
            tlx.nn.Linear(in_features=inp_size, out_features=hidden_size),
            tlx.nn.BatchNorm1d(num_features=hidden_size),
            tlx.nn.PRelu(hidden_size),
            tlx.nn.Linear(in_features=hidden_size, out_features=outp_size)
        ])

    def forward(self, x):
        return self.net(x)


class GraphEncoder(tlx.nn.Module):
    def __init__(self, feat_size,
                 projection_hidden_size,
                 projection_size):
        super().__init__()
        # TODO
        self.gnn = GCNConv(feat_size, 512)
        self.act = tlx.nn.PRelu(512)
        self.projector = MLP(512, projection_size, projection_hidden_size)

    def forward(self, feat, edge, weight, num_nodes):
        representations = self.gnn(feat, edge, weight, num_nodes)
        representations = self.act(representations)
        # representations = tlx.squeeze(representations)
        # representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


'''
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.all_weights, ma_model.all_weights):
        old_weight, up_weight = ma_params, current_params
        ma_params = ema_updater.update_average(old_weight, up_weight)
'''


def update_moving_average(ema_updater, ma_model, current_model):
    length = len(ma_model.trainable_weights)
    for i in range(length):
        # old_weight, up_weight = copy.copy(ma_model.trainable_weights[i]),  copy.copy(current_model.trainable_weights[i])
        ma_model.trainable_weights[i] = ema_updater.update_average(ma_model.trainable_weights[i], \
                                                                   current_model.trainable_weights[i])


class MERIT(tlx.nn.Module):

    def __init__(self,
                 feat_size,
                 projection_size,
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta):
        super().__init__()
        self.online_encoder = GraphEncoder(feat_size, projection_hidden_size, projection_size)
        self.target_encoder = GraphEncoder(feat_size, projection_hidden_size, projection_size)
        self.fix_weight()
        # self.target_encoder = self.online_encoder
        # set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, prediction_size, prediction_hidden_size)
        self.beta = beta

    def fix_weight(self):
        length = len(self.target_encoder.trainable_weights)
        for i in range(length):
            self.target_encoder.trainable_weights[i] = self.online_encoder.trainable_weights[i]

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def sim(self, h1, h2):
        z1 = tlx.ops.l2_normalize(h1, axis=1)
        z2 = tlx.ops.l2_normalize(h2, axis=1)
        return tlx.ops.matmul(z1, tlx.transpose(z2))

    def contrastive_loss_wo_cross_network(self, h1, h2):
        f = lambda x: tlx.exp(x)
        intra_sim = f(self.sim(h1, h1))
        inter_sim = f(self.sim(h1, h2))

        return -tlx.log(tlx.convert_to_tensor(tlx.diag(inter_sim)) / \
                        (tlx.reduce_sum(intra_sim, axis=1) + \
                         tlx.reduce_sum(inter_sim, axis=1) - \
                         tlx.convert_to_tensor(tlx.diag(intra_sim))))

    def contrastive_loss_wo_cross_view(self, h1, z):
        f = lambda x: tlx.exp(x)
        in_sim = f(self.sim(h1, h1))
        cross_sim = f(self.sim(h1, z))
        return -tlx.log(tlx.diag(cross_sim) /
                        tlx.reduce_sum(cross_sim, axis=1))

    def forward(self, feat1, edge1, weight1, num_node1, feat2, edge2, weight2, num_node2):
        online_proj_one = self.online_encoder(feat1, edge1, weight1, num_node1)
        online_proj_two = self.online_encoder(feat2, edge2, weight2, num_node2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        target_proj_one = self.target_encoder(feat1, edge1, weight1, num_node1)
        target_proj_two = self.target_encoder(feat2, edge2, weight2, num_node2)

        l1 = self.beta * self.contrastive_loss_wo_cross_network(online_pred_one, online_pred_two) + \
             (1.0 - self.beta) * self.contrastive_loss_wo_cross_view(online_pred_one, target_proj_two)
        l2 = self.beta * self.contrastive_loss_wo_cross_network(online_pred_two, online_pred_one) + \
             (1.0 - self.beta) * self.contrastive_loss_wo_cross_view(online_pred_two, target_proj_one)
        ret = (l1 + l2) / 2
        return tlx.reduce_mean(ret)

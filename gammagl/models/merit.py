import tensorlayerx as tlx
import copy
import numpy as np
import scipy.sparse as sp


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S

def calc(edge, num_node):
    weight = np.ones(edge.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(num_node, num_node))
    A = (sparse_adj + sp.eye(num_node)).tocoo()
    col, row, weight = A.col, A.row, A.data
    deg = np.array(A.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()
    return col, row, np.array(deg_inv_sqrt[row] * weight * deg_inv_sqrt[col], dtype=np.float32)



'''
class MLP(tlx.nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net_list =[]
        self.net_list.append(tlx.nn.Linear(in_features=inp_size, out_features=hidden_size))
        self.net_list.append(tlx.nn.BatchNorm1d(num_features=hidden_size))
        self.net_list.append(tlx.nn.PRelu(hidden_size,a_init=tlx.initializers.constant(0.25),data_format='channels_first')) 
        self.net_list.append(tlx.nn.Linear(in_features=hidden_size, out_features=outp_size))
        self.net=tlx.nn.Sequential(self.net_list)
        #self.linear1 = tlx.nn.Linear(out_features=hidden_size, act=None, in_features=inp_size)
        #self.bn = tlx.nn.BatchNorm1d(num_features=hidden_size)
        #self.act = tlx.nn.PRelu(hidden_size,a_init=tlx.initializers.constant(0.25))
        #self.linear3 = tlx.nn.Linear(out_features=outp_size, act=None, in_features=hidden_size)
        
    def forward(self, x):
        #z=self.linear1(x)
        #z=self.bn(z)
        #z=self.linear2(z)
        #z=self.act(z)
        #out=self.linear3(z)
        #return out
        return self.net(x)
'''
class MLP(tlx.nn.Module):
    def __init__(self, in_feat, out_feat,hid_feat):
        super(MLP, self).__init__()

        self.fc1 = tlx.nn.Linear(in_features=in_feat, out_features=hid_feat)
        #self.bn = tlx.nn.BatchNorm1d(num_features=hid_feat)
        self.fc2 = tlx.nn.Linear(in_features=hid_feat, out_features=out_feat)

    def forward(self, x):
        x = tlx.elu(self.fc1(x))
        
        return self.fc2(x)

class GraphEncoder(tlx.nn.Module):

    def __init__(self, gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn =  gnn
        self.act=tlx.nn.PRelu(512,a_init=tlx.initializers.constant(0.25))
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, feat, edge, weight, num_nodes):
        representations = self.gnn(feat, edge, weight, num_nodes)
        representations = self.act(representations)
        representations = tlx.squeeze(representations)
        #representations = representations.view(-1, representations.size(-1))
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
    length=len(np.array(ma_model.trainable_weights))
    for i in range(length):
        old_weight, up_weight = copy.copy(ma_model.trainable_weights[i]),  copy.copy(current_model.trainable_weights[i])
        ma_model.trainable_weights[i] = ema_updater.update_average(old_weight, up_weight)
#def set_requires_grad(model, val):
#    for p in model.all_weights:
#        p.requires_grad = val






class MERIT(tlx.nn.Module):
    
    def __init__(self, 
                 gnn,
                 feat_size,
                 num_layers,
                 projection_size, 
                 projection_hidden_size,
                 prediction_size,
                 activation,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta):
        
        super().__init__()

        self.online_encoder = GraphEncoder(gnn,projection_hidden_size, projection_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        #set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, prediction_size, prediction_hidden_size)
        self.beta = beta

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
    def sim(self,h1, h2):
        z1 = tlx.ops.l2_normalize(h1, axis=1)
        z2 = tlx.ops.l2_normalize(h2, axis=1)
        #return tlx.ops.matmul(z1, z2.t())
        return tlx.ops.matmul(z1,tlx.transpose(z2))

    def contrastive_loss_wo_cross_network(self,h1, h2):
        f = lambda x: tlx.exp(x)
        intra_sim = f(self.sim(h1, h1))
        inter_sim = f(self.sim(h1, h2))
        return -tlx.log(np.diag(inter_sim, k=0) /
                        ( tlx.reduce_sum(intra_sim, axis=1) + tlx.reduce_sum(inter_sim, axis=1) - np.diag(intra_sim, k=0)) )


    def contrastive_loss_wo_cross_view(self,h1,z):
        f = lambda x: tlx.exp(x)
        in_sim=f(self.sim(h1,h1))
        cross_sim = f(self.sim(h1, z))
        return -tlx.log(np.diag(cross_sim, k=0) / 
                       (tlx.reduce_sum(cross_sim, axis=1) + tlx.reduce_sum(in_sim, axis=1) - np.diag(in_sim, k=0)))
        #return -tlx.log(np.diag(cross_sim, k=0) / 
        #                tlx.reduce_sum(cross_sim, axis=1))

    def forward(self, graph1, graph2):
        online_proj_one = self.online_encoder(graph1.x, graph1.edge_index, graph1.edge_weight, graph1.num_nodes)
        online_proj_two = self.online_encoder(graph2.x, graph2.edge_index, graph2.edge_weight, graph2.num_nodes)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
                      
        #with torch.no_grad():
        target_proj_one = self.target_encoder(graph1.x, graph1.edge_index, graph1.edge_weight, graph1.num_nodes)
        target_proj_two = self.target_encoder(graph2.x, graph2.edge_index, graph2.edge_weight, graph2.num_nodes)
                       
        l1 = self.beta * self.contrastive_loss_wo_cross_network(online_pred_one, online_pred_two) + \
            (1.0 - self.beta) * self.contrastive_loss_wo_cross_view(online_pred_one, target_proj_two)
        


        l2 = self.beta * self.contrastive_loss_wo_cross_network(online_pred_two, online_pred_one) + \
            (1.0 - self.beta) * self.contrastive_loss_wo_cross_view(online_pred_two, target_proj_one)
    
        ret = (l1 + l2) / 2
        return tlx.reduce_mean(ret)
    
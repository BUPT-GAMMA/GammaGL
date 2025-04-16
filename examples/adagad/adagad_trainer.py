# from ADDataset import ADDataset
from gammagl.datasets import ADDataset
# from adagad import ADAGAD
from utils import build_args, load_best_configs, build_pre_model, build_re_model
# from adagad import to_dense_adj
import tensorlayerx as tlx
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
import numpy as np

from sklearn.metrics import roc_auc_score

def to_dense_adj(edge_index, max_num_nodes):  
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵  
    adjacency_matrix = tlx.zeros((max_num_nodes, max_num_nodes), device=edge_index.device)  
      
    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上  
    adjacency_matrix[edge_index[0], edge_index[1]] = 1  
    adjacency_matrix[edge_index[1], edge_index[0]] = 1  
      
    return adjacency_matrix

class PretrainLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(PretrainLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        x, x_, s, s_ = self.backbone_network(data['x'], data['edge_index'])
        loss = self._loss_fn(x, x_, s, s_)

        return loss

class RetrainLoss(WithLoss):
    def __init__(self, net, loss_fn,
                 alpha,
                 T,
                 loss_weight):
        super(RetrainLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.alpha = alpha
        self.T = T
        self.loss_weight = loss_weight

    def forward(self, data, y):

        x_, s_ = self.backbone_network(data['x'], data['edge_index'])
        loss = self._loss_fn(data['x'], x_, data['s'], s_, self.alpha, self.T, self.loss_weight)
        loss = loss.mean()
        return loss

def pretrain_loss(x, x_, s, s_):
    if x is None:
        return tlx.losses.mean_squared_error(s_, s)
    if s is None:
        return tlx.losses.mean_squared_error(x_, x)
    return tlx.losses.mean_squared_error(x_, x) + tlx.losses.mean_squared_error(s_, s)

def retrain_loss(x, x_, s, s_, alpha, T, loss_weight):
    # rec loss
    # score=self.rec_loss(x,x_,s,s_)
    diff_attribute = tlx.pow(x_ - x, 2)
    attribute_errors = tlx.sqrt(diff_attribute.sum(1))

    diff_structure = tlx.pow(s_ - s, 2)
    structure_errors = tlx.sqrt(diff_structure.sum(1))

    score = alpha * attribute_errors + (1 - alpha) * structure_errors

    # entropy loss
    # entropy_loss=self.log_t_entropy_loss(x,x_,s,s_,score)
    diag_s= tlx.eye(s.size()[0]).to(s.device) + s
    all_score=score.repeat(score.size()[0],1).float()

    all_score=tlx.where(diag_s.float()>0.1,all_score,tlx.convert_to_tensor(0.0, dtype=tlx.float32).to(s.device))+1e-6
    log_all_score=tlx.log(all_score) / T

    all_score = tlx.softmax(log_all_score,axis=1)

    all_log_score = -tlx.log(all_score) * all_score
    entropy_loss = all_log_score.sum(1)

    # final loss
    rank_score = score + loss_weight * entropy_loss
    return rank_score

def pretrain(net, data, max_epoch, lr, weight_decay):
    # scheduler = tlx.optimizers.lr.LambdaDecay(learning_rate=lr, lr_lambda=lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5)
    optimizer = tlx.optimizers.Adam(lr=lr, weight_decay=weight_decay)
    # metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = PretrainLoss(net, pretrain_loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in range(max_epoch):
        net.set_train()
        train_loss = train_one_step(data, data['y'])
        print("Epoch [{:0>3d}] ".format(epoch + 1) + "  train loss: {:.4f}".format(train_loss.item()))

def retrain(net, data, max_epoch, lr, weight_decay, alpha, T, loss_weight):
    net.attr_encoder.load_weights("attr_model_encoder.npz", format='npz_dict')
    net.struct_encoder.load_weights("struct_model_encoder.npz", format='npz_dict')
    net.subgraph_encoder.load_weights("subgraph_model_encoder.npz", format='npz_dict')

    for k,v in net.named_parameters():
        if k.split('.')[0]=='attr_encoder' or k.split('.')[0]=='struct_encoder' or k.split('.')[0]=='subgraph_encoder':
            v.requires_grad=False


    optimizer = tlx.optimizers.Adam(lr=lr, weight_decay=weight_decay)
    # metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = RetrainLoss(net, retrain_loss, alpha, T, loss_weight)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)


    for epoch in range(max_epoch):
        net.set_train()
        train_loss = train_one_step(data, data['y'])
        print("Epoch [{:0>3d}] ".format(epoch + 1) + "  train loss: {:.4f}".format(train_loss.item()))

def eval_func(net, data, alpha, T, loss_weight):
        x_, s_ = net(data['x'], data['edge_index'])
        score = retrain_loss(data['x'], x_, data['s'], s_, alpha, T, loss_weight)
        auc_score = roc_auc_score(tlx.convert_to_numpy(data['y'].bool()), tlx.convert_to_numpy(score))
        return auc_score

def main(args):
    # 导入数据集
    dataset = ADDataset(root='./', name=args.dataset)
    graph = dataset[0]
    # print("num_nodes: ", graph.num_nodes)
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)

    args.num_features = graph.num_features
    
    print(graph)
    print("edge_index: ", edge_index.shape)
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "num_nodes": graph.num_nodes,
    }


    # pretrain 阶段
    print('pretrain stage start...')
    attr_model, struct_model, subgraph_model = build_pre_model(args)
    
    print('======== pretrain attr encoder ========')

    pretrain(attr_model, data, args.max_epoch, args.lr, args.weight_decay)

    print('======== pretrain struct encoder ========')
    pretrain(struct_model, data, args.max_epoch, args.lr, args.weight_decay)

    print('======== pretrain subgraph encoder ========')
    pretrain(subgraph_model, data, args.max_epoch, args.lr, args.weight_decay)
    attr_model.set_eval()
    struct_model.set_eval()
    subgraph_model.set_eval()
    attr_model.encoder.save_weights("attr_model_encoder.npz", format='npz_dict')
    struct_model.encoder.save_weights("struct_model_encoder.npz", format='npz_dict')
    subgraph_model.encoder.save_weights("subgraph_model_encoder.npz", format='npz_dict')
    # retrain 阶段
    print('======== retrain stage ========')


    remodel = build_re_model(args)

    if 's' not in data:
        data['s'] = to_dense_adj(data['edge_index'], max_num_nodes=data['num_nodes'])
    if args.alpha_f is None:
        args.alpha_f = data['s'].std().detach() / (data['x'].std().detach() + data['s'].std().detach())

    retrain(remodel, data, args.max_epoch_f, args.lr_f, args.weight_decay, args.alpha_f, args.T_f, args.loss_weight_f)

    remodel.set_eval()
    remodel.save_weights("remodel.npz", format='npz_dict')
    remodel.load_weights("remodel.npz", format='npz_dict')

    print('finish train!')

    auc_score = eval_func(remodel, data, args.alpha_f, args.T_f, args.loss_weight_f)
    print("Auc: ", auc_score)



if __name__ == '__main__':
    # parameters setting
    args = build_args()
        
    if args.use_cfg:
        args = load_best_configs(args, "config_ada-gad.yml")

    if args.alpha_f=='None':
        args.alpha_f=None


    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    # args.max_epoch = 1
    # args.max_epoch_f = 1
    # args.node_encoder_num_layers = 1
    # args.edge_encoder_num_layers = 1
    # args.subgraph_encoder_num_layers = 1

    print("args: ", args)

    main(args)

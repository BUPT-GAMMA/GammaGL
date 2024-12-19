import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import tensorlayerx as tlx
from gammagl.models import FatraGNNModel
import argparse
import numpy as np
from tensorlayerx.model import TrainOneStep, WithLoss
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import yaml
from gammagl.datasets import Bail
from gammagl.datasets import Credit


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def evaluate_ged3(net, x, edge_index, y, test_mask, sens):
    net.set_eval()
    flag = 0
    output = net(x, edge_index, flag)
    pred_test = tlx.cast(tlx.squeeze(output[test_mask], axis=-1) > 0, y.dtype)

    acc_nums_test = (pred_test == y[test_mask])
    accs = np.sum(tlx.convert_to_numpy(acc_nums_test))/np.sum(tlx.convert_to_numpy(test_mask))

    auc_rocs = roc_auc_score(tlx.convert_to_numpy(y[test_mask]), tlx.convert_to_numpy(output[test_mask]))
    paritys, equalitys = fair_metric(tlx.convert_to_numpy(pred_test), tlx.convert_to_numpy(y[test_mask]), tlx.convert_to_numpy(sens[test_mask]))

    return accs, auc_rocs, paritys, equalitys


class DicLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(DicLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        output = self.backbone_network(data['x'], data['edge_index'], data['flag'])
        loss = tlx.losses.binary_cross_entropy(tlx.squeeze(output, axis=-1), tlx.cast(data['sens'], dtype=tlx.float32))
        return loss


class EncClaLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(EncClaLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        output = self.backbone_network(data['x'], data['edge_index'], data['flag'])
        y_train = tlx.cast(tlx.expand_dims(label[data['train_mask']], axis=1), dtype=tlx.float32)
        loss = tlx.losses.binary_cross_entropy(output[data['train_mask']], y_train)
        return loss
    

class EncLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(EncLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        output = self.backbone_network(data['x'], data['edge_index'], data['flag'])
        loss = tlx.losses.mean_squared_error(output, 0.5 * tlx.ones_like(output))
        return loss
    
    
class EdtLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(EdtLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        output = self.backbone_network(data['x'], data['edge_index'], data['flag'])
        loss = -tlx.abs(tlx.reduce_sum(output[data['train_mask']][data['t_idx_s0_y1']])) / tlx.reduce_sum(tlx.cast(data['t_idx_s0_y1'], dtype=tlx.float32)) - tlx.reduce_sum(output[data['train_mask']][data['t_idx_s1_y1']]) / tlx.reduce_sum(tlx.cast(data['t_idx_s1_y1'], dtype=tlx.float32))

        return loss
    

class AliLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(AliLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        output = self.backbone_network(data['x'], data['edge_index'], data['flag'])
        h1 = output['h1']
        h2 = output['h2']
        idx_s0_y0 = data['idx_s0_y0']
        idx_s1_y0 = data['idx_s1_y0']
        idx_s0_y1 = data['idx_s0_y1']
        idx_s1_y1 = data['idx_s1_y1']
        node_num = data['x'].shape[0]
        loss_align = - node_num / (tlx.reduce_sum(tlx.cast(idx_s0_y0, dtype=tlx.float32)))  * tlx.reduce_mean(tlx.matmul(h1[idx_s0_y0], tlx.transpose(h2[idx_s0_y0]))) \
                        - node_num / (tlx.reduce_sum(tlx.cast(idx_s0_y1, dtype=tlx.float32))) * tlx.reduce_mean(tlx.matmul(h1[idx_s0_y1], tlx.transpose(h2[idx_s0_y1]))) \
                        - node_num / (tlx.reduce_sum(tlx.cast(idx_s1_y0, dtype=tlx.float32))) * tlx.reduce_mean(tlx.matmul(h1[idx_s1_y0], tlx.transpose(h2[idx_s1_y0]))) \
                        - node_num / (tlx.reduce_sum(tlx.cast(idx_s1_y1, dtype=tlx.float32))) * tlx.reduce_mean(tlx.matmul(h1[idx_s1_y1], tlx.transpose(h2[idx_s1_y1])))

        loss = loss_align * 0.01       
        return loss


def main(args):

    # load datasets
    if str.lower(args.dataset) not in ['bail', 'credit', 'pokec']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    
    if args.dataset == 'bail':
        dataset = Bail(args.dataset_path, args.dataset)
    
    elif args.dataset == 'credit':
        dataset = Credit(args.dataset_path, args.dataset)
    
    graphs = dataset.data
    data = {
        'x':graphs[0].x,
        'y': graphs[0].y,
        'edge_index': {'edge_index': graphs[0].edge_index},
        'sens': graphs[0].sens,
        'train_mask': graphs[0].train_mask,
    }
    data_test = []
    for i in range(1, len(graphs)):
        data_tem = {
            'x':graphs[i].x,
            'y': graphs[i].y,
            'edge_index': graphs[i].edge_index,
            'sens': graphs[i].sens,
            'test_mask': graphs[i].train_mask | graphs[i].val_mask | graphs[i].test_mask,
        }
        data_test.append(data_tem)
    dataset = None
    graphs = None
    args.num_features, args.num_classes = data['x'].shape[1], len(np.unique(tlx.convert_to_numpy(data['y']))) - 1
    args.test_set_num = len(data_test)

    t_idx_s0 = data['sens'][data['train_mask']] == 0
    t_idx_s1 = data['sens'][data['train_mask']] == 1
    t_idx_s0_y1 = tlx.logical_and(t_idx_s0, data['y'][data['train_mask']] == 1)
    t_idx_s1_y1 = tlx.logical_and(t_idx_s1, data['y'][data['train_mask']] == 1)
 
    idx_s0 = data['sens'] == 0
    idx_s1 = data['sens'] == 1
    idx_s0_y1 = tlx.logical_and(idx_s0, data['y'] == 1)
    idx_s1_y1 = tlx.logical_and(idx_s1, data['y'] == 1)
    idx_s0_y0 = tlx.logical_and(idx_s0, data['y'] == 0)
    idx_s1_y0 = tlx.logical_and(idx_s1, data['y'] == 0)

    data['idx_s0_y0'] = idx_s0_y0
    data['idx_s1_y0'] = idx_s1_y0
    data['idx_s0_y1'] = idx_s0_y1 
    data['idx_s1_y1'] = idx_s1_y1 
    data['t_idx_s0_y1'] = t_idx_s0_y1 
    data['t_idx_s1_y1'] = t_idx_s1_y1

    edge_index_np = tlx.convert_to_numpy(data['edge_index']['edge_index'])
    adj = sp.coo_matrix((np.ones(data['edge_index']['edge_index'].shape[1]), (edge_index_np[0, :], edge_index_np[1, :])),
                        shape=(data['x'].shape[0], data['x'].shape[0]),
                        dtype=np.float32)
    A2 = adj.dot(adj)
    A2 = A2.toarray()
    A2_edge = tlx.convert_to_tensor(np.vstack((A2.nonzero()[0], A2.nonzero()[1])))

    net = FatraGNNModel(args)

    dic_loss_func = DicLoss(net, tlx.losses.binary_cross_entropy)
    enc_cla_loss_func = EncClaLoss(net, tlx.losses.binary_cross_entropy)
    enc_loss_func = EncLoss(net, tlx.losses.binary_cross_entropy)
    edt_loss_func = EdtLoss(net, tlx.losses.binary_cross_entropy)
    ali_loss_func = AliLoss(net, tlx.losses.binary_cross_entropy)

    dic_opt = tlx.optimizers.Adam(lr=args.d_lr, weight_decay=args.d_wd)
    dic_train_one_step = TrainOneStep(dic_loss_func, dic_opt, net.discriminator.trainable_weights)

    enc_cla_opt = tlx.optimizers.Adam(lr=args.c_lr, weight_decay=args.c_wd)
    enc_cla_train_one_step = TrainOneStep(enc_cla_loss_func, enc_cla_opt, net.encoder.trainable_weights+net.classifier.trainable_weights)

    enc_opt = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd)
    enc_train_one_step = TrainOneStep(enc_loss_func, enc_opt, net.encoder.trainable_weights)

    edt_opt = tlx.optimizers.Adam(lr=args.g_lr, weight_decay=args.g_wd)
    edt_train_one_step = TrainOneStep(edt_loss_func, edt_opt, net.graphEdit.trainable_weights)

    ali_opt = tlx.optimizers.Adam(lr=args.e_lr, weight_decay=args.e_wd)
    ali_train_one_step = TrainOneStep(ali_loss_func, ali_opt, net.encoder.trainable_weights)

    tlx.set_seed(args.seed)
    net.set_train()
    for epoch in range(0, args.epochs):
        print(f"======={epoch}=======")
        # train discriminator to recognize the sensitive group
        data['flag'] = 1
        for epoch_d in range(0, args.dic_epochs):
            dic_loss = dic_train_one_step(data=data, label=data['y'])

        # train classifier and encoder
        data['flag'] = 2
        for epoch_c in range(0, args.cla_epochs):
            enc_cla_loss = enc_cla_train_one_step(data=data, label=data['y'])

        # train encoder to fool discriminator
        data['flag'] = 3
        for epoch_g in range(0, args.g_epochs):
            enc_loss = enc_train_one_step(data=data, label=data['y'])

        # train generator
        data['flag'] = 4
        if epoch > args.start:
            if epoch % 10 == 0:
                if epoch % 20 == 0:
                    data['edge_index']['edge_index2'] = net.graphEdit.modify_structure1(data['edge_index']['edge_index'], A2_edge, data['sens'], data['x'].shape[0], args.drope_rate)
                else:
                    data['edge_index']['edge_index2'] = net.graphEdit.modify_structure2(data['edge_index']['edge_index'], A2_edge, data['sens'], data['x'].shape[0], args.drope_rate)
            else:
                data['edge_index']['edge_index2'] = data['edge_index']['edge_index']
                
            for epoch_g in range(0, args.dtb_epochs):
                edt_loss = edt_train_one_step(data=data, label=data['y'])

        # shift align
        data['flag'] = 5
        if epoch > args.start:
            for epoch_a in range(0, args.a_epochs):
                aliloss = ali_train_one_step(data=data, label=data['y'])

    acc = np.zeros([args.test_set_num])
    auc_roc = np.zeros([args.test_set_num])
    parity = np.zeros([args.test_set_num])
    equality = np.zeros([args.test_set_num])
    net.set_eval()
    for i in range(args.test_set_num):
        data_tem = data_test[i]
        acc[i],auc_roc[i], parity[i], equality[i] = evaluate_ged3(net, data_tem['x'], data_tem['edge_index'], data_tem['y'], data_tem['test_mask'], data_tem['sens'])
    return acc, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bail')
    parser.add_argument('--start', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--dic_epochs', type=int, default=5)
    parser.add_argument('--dtb_epochs', type=int, default=5)
    parser.add_argument('--cla_epochs', type=int, default=12)
    parser.add_argument('--a_epochs', type=int, default=2)
    parser.add_argument('--g_epochs', type=int, default=5)
    parser.add_argument('--g_lr', type=float, default=0.05)
    parser.add_argument('--g_wd', type=float, default=0.01)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.001)
    parser.add_argument('--c_wd', type=float, default=0.01)
    parser.add_argument('--e_lr', type=float, default=0.005)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--drope_rate', type=float, default=0.1)
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    
    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
    args.device = f'cuda:{args.gpu}'


    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'config.yaml')
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        config_dict = yaml.safe_load(cont)[args.dataset]
    for key, value in config_dict.items():
        args.__setattr__(key, value)

    print(args)
    acc, auc_roc, parity, equality = main(args)

    for i in range(args.test_set_num):
        print("===========test{}============".format(i+1))
        print('Acc: ', acc.T[i])
        print('auc_roc: ', auc_roc.T[i])
        print('parity: ', parity.T[i])
        print('equality: ', equality.T[i])

import os
import os.path as osp
# os.environ['TL_BACKEND'] = 'paddle' # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.insert(0, osp.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse

from tqdm import tqdm
import numpy as np
import tensorlayerx as tlx
from gammagl.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon # , Actor
from gammagl.models import GPRGNNModel
from gammagl.utils.loop import add_self_loops
from gammagl.utils import calc_gcn_norm
from tensorlayerx.model import TrainOneStep, WithLoss
import gammagl.transforms as T


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        if tlx.BACKEND == 'mindspore':
            idx = tlx.convert_to_tensor([i for i, v in enumerate(data['train_mask']) if v], dtype=tlx.int64)
            train_logits = tlx.gather(logits,idx)
            train_label = tlx.gather(label,idx)
        else:
            train_logits = logits[data['train_mask']]
            train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        return loss


def evaluate(net, data, y, mask, metrics):
    net.set_eval()
    logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    if tlx.BACKEND == 'mindspore':
        idx = tlx.convert_to_tensor([i for i, v in enumerate(mask) if v],dtype=tlx.int64)
        _logits = tlx.gather(logits,idx)
        _label = tlx.gather(y,idx)
    else:
        _logits = logits[mask]
        _label = y[mask]
    metrics.update(_logits, _label)
    acc = metrics.result()
    metrics.reset()
    return acc


def index_to_mask(index, size):
    mask = np.zeros(size, dtype=np.bool_)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero()[0]
        np.random.shuffle(index)
        indices.append(index)

    train_index = np.concatenate([i[:percls_trn] for i in indices])

    if Flag == 0:
        rest_index = np.concatenate([i[percls_trn:] for i in indices])
        np.random.shuffle(rest_index)
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = np.concatenate([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = np.concatenate([i[percls_trn+val_lb:] for i in indices])
        np.random.shuffle(rest_index)
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data




def main(args):
    # load datasets
    if str.lower(args.dataset) in ['cora','pubmed','citeseer']:
        dataset = Planetoid(args.dataset_path, args.dataset, transform=T.NormalizeFeatures())
        dataset.process()
        graph = dataset[0]
    elif str.lower(args.dataset) in ['cornell', 'texas']:
        dataset = WebKB(args.dataset_path, args.dataset)
        dataset.process()
        graph = dataset[0]
    elif str.lower(args.dataset) in ['computers', 'photo']:
        dataset = Amazon(
            root=args.dataset_path, name=args.dataset, transform=T.NormalizeFeatures())
        dataset.process()
        graph = dataset[0]
    elif str.lower(args.dataset) in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root=args.dataset_path, name=args.dataset, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=args.dataset_path, name=args.dataset, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        preProcDs.process()
        dataset.process()
        graph = dataset[0]
        graph.edge_index = preProcDs[0].edge_index
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
        
    graph.numpy()
    graph.num_nodes = graph.x.shape[0]

###########
####  split the datasets as defined in GPRGNN original paper
###########
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(graph.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(graph.y)))
    data = random_planetoid_splits(graph, dataset.num_classes, percls_trn, val_lb)

    graph.tensor()
    edge_index, _ = add_self_loops(graph.edge_index, n_loops=args.self_loops)
    edge_weight = tlx.ops.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))
    x = graph.x
    y = graph.y

    net = GPRGNNModel(feature_dim=x.shape[1],
                   hidden_dim=args.hidden_dim,
                   num_class=dataset.num_classes,
                   drop_rate=args.drop_rate,
                   K=args.K,
                   Init=args.Init,
                   alpha=args.alpha,
                   dprate=args.dprate,
                   Gamma=args.Gamma)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights
    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    
    data = {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = test_acc = 0
    val_acc_history = []

    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        train_loss = train_one_step(data, y)
        val_acc = evaluate(net, data, graph.y, data['val_mask'], metrics)
        val_acc_history.append(val_acc)

        # print("Epoch [{:0>3d}] ".format(epoch+1)\
        #       + "  train loss: {:.4f}".format(train_loss.item())\
        #       + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+net.name+'_'+args.dataset+".npz", format='npz_dict')
        
        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = np.array(val_acc_history[-(args.early_stopping + 1):-1])
            tmp = tlx.convert_to_tensor(tmp.mean())
            if val_acc < tmp:
                break

    net.load_weights(args.best_model_path+net.name+'_'+args.dataset+".npz", format='npz_dict')
    test_acc = evaluate(net, data, graph.y, data['test_mask'], metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    print("learnable weight:{}".format(tlx.convert_to_numpy(net.all_weights[-1])))
    record_path = osp.abspath("./test_accuracy")
    
    with open(record_path, "a+") as f:
        f.write("{} {} {:.2f}\n".format(tlx.BACKEND, args.dataset, test_acc * 100))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.05, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=1000, help="number of epoch")
    parser.add_argument("--early_stopping", type=int, default=200, help="epoch begining to early stop")
    parser.add_argument("--hidden_dim", type=int, default=64, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-3, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../datasets', help="path to save dataset")
    parser.add_argument("--train_rate", type=float, default=0.6, help="ratio of training set")
    parser.add_argument("--val_rate", type=float, default=0.2, help="ratio of validation set")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--dprate", type=float, default=0.5, help="drop rate of gprprop")
    parser.add_argument("--Init", type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default="PPR", help="initializaiton method of learnable weight of gprprop")
    parser.add_argument("--K", type=int, default=10, help="depth of gprprop")
    parser.add_argument("--alpha", type=float, default=0.2, help="initialization of learnable weight of gprprop")
    parser.add_argument("--Gamma", type=int)
    args = parser.parse_args()

    main(args)




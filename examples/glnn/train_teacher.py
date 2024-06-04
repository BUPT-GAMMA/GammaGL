# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import yaml
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid, Amazon
from gammagl.models import GCNModel, GraphSAGE_Full_Model, GATModel, APPNPModel, MLP
from gammagl.utils import mask_to_index, calc_gcn_norm
from tensorlayerx.model import TrainOneStep, WithLoss


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        if args.teacher == "GCN":
            logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        elif args.teacher == "SAGE":
            logits = self.backbone_network(data['x'], data['edge_index'])
        elif args.teacher == "GAT":
            logits = self.backbone_network(data['x'], data['edge_index'], data['num_nodes'])
        elif args.teacher == "APPNP":
            logits = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        elif args.teacher == "MLP":
            logits = self.backbone_network(data['x'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def get_training_config(config_path, model_name, dataset):
    with open(config_path, "r") as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = full_config["global"]
    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config

    specific_config["model_name"] = model_name
    return specific_config


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def train_teacher(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer','computers','photo']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    if args.dataset in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(args.dataset_path, args.dataset)
    elif args.dataset == 'computers':
        dataset = Amazon(args.dataset_path, args.dataset, train_ratio=200/13752, val_ratio=(200/13752)*1.5)
    elif args.dataset == 'photo':
        dataset = Amazon(args.dataset_path, args.dataset, train_ratio=160/7650, val_ratio=(160/7650)*1.5)
    graph = dataset[0]
    edge_index = graph.edge_index
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    if args.teacher == "GCN":
        net = GCNModel(feature_dim=dataset.num_node_features, 
                       hidden_dim=conf["hidden_dim"], 
                       num_class=dataset.num_classes,
                       drop_rate=conf["dropout_ratio"], 
                       num_layers=conf["num_layers"])
        
    elif args.teacher == "SAGE":
        net = GraphSAGE_Full_Model(in_feats=dataset.num_node_features, 
                                   n_hidden=conf["hidden_dim"], 
                                   n_classes=dataset.num_classes,
                                   n_layers=conf["num_layers"], 
                                   activation=tlx.nn.ReLU(), 
                                   dropout=conf["dropout_ratio"], 
                                   aggregator_type='gcn')
        
    elif args.teacher == "GAT":
        net = GATModel(feature_dim=dataset.num_node_features, 
                       hidden_dim=conf["hidden_dim"], 
                       num_class=dataset.num_classes,
                       heads=conf["num_heads"], 
                       drop_rate=conf["dropout_ratio"], 
                       num_layers=conf["num_layers"])
        
    elif args.teacher == "APPNP":
        net = APPNPModel(feature_dim=dataset.num_node_features, 
                         num_class=dataset.num_classes,
                         iter_K=10, 
                         alpha=0.1, 
                         drop_rate=conf["dropout_ratio"])
        
    elif args.teacher == "MLP":
        net = MLP(in_channels=dataset.num_node_features,
                  hidden_channels=conf["hidden_dim"],
                  out_channels=dataset.num_classes,
                  num_layers=conf["num_layers"],
                  act=tlx.nn.ReLU(),
                  dropout=conf["dropout_ratio"])

    optimizer = tlx.optimizers.Adam(lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes
    } 

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        if args.teacher == "GCN":
            logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
        elif args.teacher == "SAGE":
            logits = net(data['x'], data['edge_index'])
        elif args.teacher == "GAT":
            logits = net(data['x'], data['edge_index'], data['num_nodes'])
        elif args.teacher == "APPNP":
            logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        elif args.teacher == "MLP":
            logits = net(data['x'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+args.dataset+'_'+args.teacher+".npz", format='npz_dict')
            if tlx.BACKEND == 'torch':
                tlx.files.save_any_to_npy(tlx.convert_to_numpy(logits), args.best_model_path+args.dataset+'_'+args.teacher+'_logits.npy')
            else:
                tlx.files.save_any_to_npy(tlx.convert_to_numpy(logits), args.best_model_path+args.dataset+'_'+args.teacher+'_logits.npy')

    net.load_weights(args.best_model_path+args.dataset+'_'+args.teacher+".npz", format='npz_dict')
    net.set_eval()
    if args.teacher == "GCN":
        logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
    elif args.teacher == "SAGE":
        logits = net(data['x'], data['edge_index'])
    elif args.teacher == "GAT":
        logits = net(data['x'], data['edge_index'], data['num_nodes'])
    elif args.teacher == "APPNP":
        logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    elif args.teacher == "MLP":
        logits = net(data['x'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default="./train.conf.yaml",help="path to modelconfigeration")
    parser.add_argument("--teacher", type=str, default="SAGE", help="teacher model")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument('--dataset', type=str, default="cora", help="dataset")
    parser.add_argument("--dataset_path", type=str, default=r'./data', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, args.teacher, args.dataset)
    conf = dict(args.__dict__, **conf)
    
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    train_teacher(args)

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
from gammagl.models import MLP
from gammagl.utils import mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, teacher_logits):
        student_logits = self.backbone_network(data['x'])
        train_y = tlx.gather(data['y'], data['t_idx'])
        train_teacher_logits = tlx.gather(teacher_logits, data['t_idx'])
        train_student_logits = tlx.gather(student_logits, data['t_idx'])
        loss = self._loss_fn(train_y, train_student_logits, train_teacher_logits, args.lamb)
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


def kl_divergence(teacher_logits, student_logits):  
    # convert logits to probabilities
    teacher_probs = tlx.softmax(teacher_logits)
    student_probs = tlx.softmax(student_logits)
    # compute KL divergence
    kl_div = tlx.reduce_sum(teacher_probs * (tlx.log(teacher_probs+1e-10) - tlx.log(student_probs+1e-10)), axis=-1)
    return tlx.reduce_mean(kl_div)


def cal_mlp_loss(labels, student_logits, teacher_logits, lamb):
    loss_l = tlx.losses.softmax_cross_entropy_with_logits(student_logits, labels)
    loss_t = kl_divergence(teacher_logits, student_logits)
    return lamb * loss_l + (1 - lamb) * loss_t


def train_student(args):
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

    # load teacher_logits from .npy file
    teacher_logits = tlx.files.load_npy_to_any(path = r'./', name = f'{args.dataset}_{args.teacher}_logits.npy')
    teacher_logits = tlx.ops.convert_to_tensor(teacher_logits)

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)
    t_idx = tlx.concat([train_idx, test_idx, val_idx], axis=0)

    net = MLP(in_channels=dataset.num_node_features,
              hidden_channels=conf["hidden_dim"],
              out_channels=dataset.num_classes,
              num_layers=conf["num_layers"],
              act=tlx.nn.ReLU(),
              norm=None,
              dropout=float(conf["dropout_ratio"]))

    optimizer = tlx.optimizers.Adam(lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, cal_mlp_loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "t_idx": t_idx
    } 

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, teacher_logits)
        net.set_eval()
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
            net.save_weights(args.best_model_path+args.dataset+"_"+args.teacher+"_MLP.npz", format='npz_dict')

    net.load_weights(args.best_model_path+args.dataset+"_"+args.teacher+"_MLP.npz", format='npz_dict')
    net.set_eval()
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
    parser.add_argument("--lamb", type=float, default=0, help="parameter balances loss from hard labels and teacher outputs")
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

    train_student(args)

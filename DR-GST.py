import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TL_BACKEND'] = 'torch'
#tensorlayerx\optimizers\torch_optimizers.py 264行 loss.bakcward() -> loss.backward(retain_graph=True)
import argparse
import tensorlayerx as tlx
import torch
import tensorflow as tf
from tensorlayerx.model import TrainOneStep, WithLoss
from tensorlayerx.losses import mean_squared_error
from gammagl.datasets import Planetoid
from gammagl.datasets import Amazon
from gammagl.datasets import Coauthor
from gammagl.models import GCNModel, GATModel, GraphSAGE_Full_Model, APPNPModel, GINModel, SGCModel, MIXHOPModel
from gammagl.utils import add_self_loops, mask_to_index
from gammagl.data import Dataset
from gammagl.transforms import DropEdge
import numpy as np
import random


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


class MyWithLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(MyWithLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self.weighted_cross_entropy(train_logits, train_y, data['bald'][data['train_idx']], data["beta"],
                                           data["num_class"], data["T"])
        return loss

    def weighted_cross_entropy(self, output, labels, bald, beta, num_class, T):
        bald += 1e-6
        output = tlx.softmax(output, axis=1)
        output = tlx.matmul(output, T)
        bald = bald / (tlx.reduce_mean(bald) * beta)
        labels = self.one_hot(labels, num_class)
        tmp = tlx.reduce_sum(output * labels, axis=1)
        loss = -tlx.log(tmp)
        loss = tlx.reduce_sum(loss * bald)
        loss /= labels.size()[0]
        return loss

    @staticmethod
    def one_hot(labels, num_class):
        tensor = tlx.zeros((labels.size()[0], num_class))
        for i in range(labels.size()[0]):
            tensor[i][labels[i]] = 1
        return tensor


class DR_GST():
    @staticmethod
    def generate_mask(dataset, labels, labelrate, os_path=None):
        datalength = labels.size()
        train_mask, val_mask, test_mask = tlx.ones_like((1, datalength[0]), dtype=bool), tlx.zeros_like(
            (1, datalength[0]), dtype=bool) \
            , tlx.ones_like((1, datalength[0]), dtype=bool)
        path = 'data/%s/' % (dataset)
        if os_path != None:
            path = os_path + '/' + path
        mask = (train_mask, val_mask, test_mask)
        name = ('train', 'val', 'test')
        for (i, na) in enumerate(name):
            with open(path + na + '%s.txt' % labelrate, 'r') as f:
                index = f.read().splitlines()
                index = list(map(int, index))
                mask[i][0][index] = 1
        return mask[0][0], mask[1][0], mask[2][0]

    @staticmethod
    def update_T(T, output, idx_train, labels):
        if tlx.BACKEND == 'tensorflow':  # TODO 只有两个版本
            output = tf.keras.activations.softmax(output, axis=1)
            T.requires_grad = True
            index = mask_to_index(idx_train)
            nclass = labels.max().item() + 1
            optimizer = tf.optimizers.Adam(learning_rate=0.01, weight_decay=5e-4)
            for epoch in range(200):
                with tf.GradientTape() as tape:
                    loss = mean_squared_error(output[index], T[labels[index]]) + mean_squared_error(T, tlx.eye(nclass))
                grads = tape.gradient(loss, T)
                optimizer.apply_gradients(zip(grads, [T]))
        elif tlx.BACKEND == 'torch':
            output = torch.softmax(output, dim=1)
            T.requires_grad = True
            optimizer = torch.optim.Adam([T], lr=0.01, weight_decay=5e-4)
            mse_criterion = torch.nn.MSELoss()
            index = torch.where(idx_train)[0]
            nclass = labels.max().item() + 1
            for epoch in range(200):
                optimizer.zero_grad()
                loss = mse_criterion(output[index].detach(), T[labels[index]].clone()) + mse_criterion(T.clone(), torch.eye(nclass))
                loss.backward()
                optimizer.step()
            T.requires_grad = False
        else:
            raise NotImplementedError('This backend is not implemented')
        return T

    @staticmethod
    def calculate_acc(logits, y, metrics):
        metrics.update(logits, y)
        rst = metrics.result()
        metrics.reset()
        return rst

    @staticmethod
    def generate_trainmask(dataset, labels, labelrate):
        data = dataset.data
        train_mask = data.train_mask
        nclass = dataset.num_classes
        train_index = mask_to_index(train_mask)
        train_mask = train_mask.clone()
        train_mask[:] = False
        label = labels[train_index]
        for i in range(nclass):
            class_index = mask_to_index(label == i).tolist()
            class_index = random.sample(class_index, labelrate)
            train_mask[train_index[class_index]] = True
        return train_mask

    @staticmethod
    def train(graph, args, bald, T, idx_val, idx_train, pseudo_labels):
        net = DR_GST.get_model(args, graph)
        data = graph.data
        best, bad_counter = 100, 0
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes, n_loops=args.self_loops)

        train_idx = idx_train
        test_idx = mask_to_index(data.test_mask)
        val_idx = mask_to_index(data.val_mask)

        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
        train_weights = net.trainable_weights

        net_with_loss = MyWithLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
        train_one_step = TrainOneStep(net_with_loss, optimizer, train_weights)
        metrics = tlx.metrics.Accuracy()

        train_data = {
            "x": data.x,
            "y": data.y,
            "edge_index": edge_index,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "val_idx": val_idx,
            "num_nodes": data.num_nodes,
            "num_class": graph.num_classes,
            "bald": bald,
            "T": T,
            "beta": args.beta
        }

        best_val_acc = 0
        for epoch in range(args.epochs):
            net.set_train()
            train_loss = train_one_step(train_data, data.y.detach())

            net.set_eval()
            logits = net(train_data['x'], train_data['edge_index'], None, train_data['num_nodes'])
            val_logits = tlx.gather(logits.clone(), train_data['val_idx'])
            val_y = tlx.gather(train_data['y'], train_data['val_idx'])
            val_acc = DR_GST.calculate_acc(val_logits, val_y, metrics)

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(train_loss.item()) \
                  + "  val acc: {:.4f}".format(val_acc))

            one_hot_labels = MyWithLoss.one_hot(pseudo_labels[idx_val], train_data['num_class'])
            loss_val = tlx.losses.softmax_cross_entropy_with_logits(logits[idx_val], one_hot_labels)

            model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
                args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)

            if loss_val < best:
                tlx.files.save_npz(net.all_weights, name=model_path + net.name + ".npz")
                best = loss_val
                bad_counter = 0
                best_output = logits
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

        return best_output

    @staticmethod
    def regenerate_pseudo_label(output, labels, idx_train, unlabeled_index, threshold, sign=False):

        're-generate pseudo labels every stage'
        unlabeled_index = mask_to_index(unlabeled_index == True)
        confidence, pred_label = DR_GST.get_confidence(output, sign)
        index = mask_to_index(confidence > threshold)
        pseudo_index = []
        pseudo_labels, idx_train_ag = labels.detach(), idx_train.detach()
        for i in index:
            if i not in idx_train:
                pseudo_labels[i] = pred_label[i]
                # pseudo_labels[i] = labels[i]
                if i in unlabeled_index:
                    idx_train_ag[i] = True
                    pseudo_index.append(i)
        idx_pseudo = tlx.zeros_like(idx_train)
        pseudo_index = tlx.convert_to_tensor(pseudo_index)
        if pseudo_index.size()[0] != 0:
            idx_pseudo[pseudo_index] = 1
        return idx_train_ag, pseudo_labels, idx_pseudo

    @staticmethod
    def get_confidence(output, with_softmax=False):
        if not with_softmax:
            output = tlx.softmax(output, axis=1)

        confidence = tlx.reduce_max(output, axis=1)
        pred_label = tlx.argmax(output, axis=1)

        return confidence, pred_label

    @staticmethod
    def uncertainty_dropedge(droped_data, model_path, args, dataset):
        f_pass = 100
        net = DR_GST.get_model(args, dataset)
        tlx.files.load_and_assign_npz(name=model_path + net.name + ".npz", network=net)
        out_list = []
        net.set_eval()
        for _ in range(f_pass):
            output = net(droped_data.x, droped_data.edge_index, None, droped_data.num_nodes)
            output = tlx.softmax(output, axis=1)
            out_list.append(output)
        out_list = tlx.stack(out_list)
        out_mean = tlx.reduce_mean(out_list, axis=0)
        entropy = tlx.reduce_sum(tlx.reduce_mean(out_list * tlx.log(out_list), axis=0), axis=1)
        Eentropy = tlx.reduce_sum(out_mean * tlx.log(out_mean), axis=1)
        bald = entropy - Eentropy
        return bald

    @staticmethod
    def uncertainty_dropout(model_path, args, dataset):
        f_pass = 100
        net = DR_GST.get_model(args, dataset)
        tlx.files.load_and_assign_npz(name=model_path + net.name + ".npz", network=net)
        net.set_eval()
        out_list = []
        data = dataset.data
        for _ in range(f_pass):
            output = net(data.x, data.edge_index, None, data.num_nodes)
            output = tlx.softmax(output, axis=1)
            out_list.append(output)
        out_list = tlx.stack(out_list)
        out_mean = tlx.reduce_mean(out_list, axis=0)
        entropy = tlx.reduce_sum(tlx.reduce_mean(out_list * tlx.log(out_list), axis=0), axis=1)
        Eentropy = tlx.reduce_sum(out_mean * tlx.log(out_mean), axis=1)
        bald = entropy - Eentropy

        return bald

    @staticmethod
    def load_dataset(model_path, name):
        if name == 'Cora':
            dataset = Planetoid(model_path, name)
        elif name == 'Citeseer':
            dataset = Planetoid(model_path, name)
        elif name == 'Pubmed':
            dataset = Planetoid(model_path, name)
        elif name == 'CS':
            dataset = Coauthor(model_path, "CS")
        elif name == 'Physics':
            dataset = Coauthor(model_path, "Physics")
        elif name == 'Computers':
            dataset = Amazon(model_path, "Computers")
        elif name == 'Photo':
            dataset = Amazon(model_path, "Photo")
        else:
            dataset = Dataset()
            dataset.load_data(model_path)
        return dataset

    @staticmethod
    def get_model(args, dataset, sign=True):
        model_name = args.model
        if sign:
            droprate = args.dropout
        else:
            droprate = args.droprate
        if model_name == 'GCN':
            model = GCNModel(feature_dim=dataset.num_node_features,
                             hidden_dim=args.hidden_dim,
                             num_class=dataset.num_classes,
                             drop_rate=args.droprate,
                             num_layers=args.num_layers,
                             norm=args.norm,
                             name="GCN")
        elif model_name == 'GAT':
            model = GATModel(feature_dim=dataset.num_node_features,
                             hidden_dim=args.hidden_dim,
                             num_class=dataset.num_classes,
                             heads=([args.nb_heads] * 1) + [args.nb_out_heads],
                             drop_rate=args.droprate,
                             num_layers=args.num_layers,
                             name="GAT")
        elif model_name == 'GraphSAGE':
            model = GraphSAGE_Full_Model(in_feats=dataset.num_node_features,
                                         n_hidden=args.hidden,
                                         n_classes=dataset.num_classes,
                                         n_layers=args.num_layers,
                                         activation=tlx.nn.ReLU(),
                                         dropout=droprate,
                                         aggregator_type='gcn')
        elif model_name == 'APPNP':
            model = APPNPModel(feature_dim=dataset.num_node_features,
                               num_class=dataset.num_classes,
                               iter_K=10,
                               alpha=0.1,
                               drop_rate=droprate,
                               name="APPNP")
        elif model_name == 'GIN':
            model = GINModel(in_channels=dataset.num_node_features,
                             hidden_channels=args.hidden,
                             out_channels=dataset.num_classes,
                             num_layers=args.num_layers)
        elif model_name == 'SGC':
            model = SGCModel(feature_dim=dataset.num_node_features,
                             num_class=dataset.num_classes,
                             iter_K=2)
        elif model_name == 'MixHop':
            model = MIXHOPModel(feature_dim=dataset.num_node_features,
                                hidden_dim=args.hidden,
                                out_dim=dataset.num_classes,
                                p=[0, 1, 2],
                                drop_rate=droprate,
                                num_layers=args.num_layers,
                                norm=True,
                                name="MixHop")

        return model


def main(args):
    model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
        args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)

    dataset_name = args.dataset
    dataset = DR_GST.load_dataset(model_path, dataset_name)

    data = dataset.data
    labels = data.y
    num_node = data.num_nodes
    num_class = dataset.num_classes

    if dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    else:
        idx_train, idx_val, idx_test = DR_GST.generate_mask(dataset=dataset_name, labels=labels, labelrate=20)

    train_index = mask_to_index(idx_train)
    idx_train_ag = mask_to_index(data.train_mask)
    if tlx.BACKEND == 'torch':
        pseudo_labels = labels.clone()  # TODO tensorflow 会报错
    elif tlx.BACKEND == 'tensorflow':
        pseudo_labels = tf.optimizers.copy(labels)
    bald = tlx.ones((num_node,))
    T = tlx.ops.eye(num_class, num_class)
    T.requires_grad = False

    if args.drop_method == 'dropedge':
        drop_edge_data = DropEdge(p=0.1)(data)

    if args.labelrate != 20:
        idx_train[train_index] = True
        idx_train = DR_GST.generate_trainmask(dataset, labels, args.labelrate)

    for s in range(args.stage):
        best_output = DR_GST.train(dataset, args, bald, T, idx_val, idx_train_ag, pseudo_labels)
        T = DR_GST.update_T(T, best_output, idx_train, data.y.data)
        idx_unlabeled = ~(idx_train | idx_test | idx_val)

        if args.drop_method == 'dropout':
            bald = DR_GST.uncertainty_dropout(model_path, args, dataset)
        elif args.drop_method == 'dropedge':
            bald = DR_GST.uncertainty_dropedge(drop_edge_data, model_path, args, dataset)

        net = DR_GST.get_model(args, dataset)
        tlx.files.load_and_assign_npz(name=model_path + net.name + ".npz", network=net)
        if tlx.BACKEND == 'torch':
            net.to(data['x'].device)
        net.set_eval()
        best_output = net(data.x, data.edge_index, None, data.num_nodes)
        idx_train_ag, pseudo_labels, idx_pseudo = DR_GST.regenerate_pseudo_label(best_output, labels, idx_train,idx_unlabeled, args.threshold)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='dataset for training')
    parser.add_argument("--dataset_path", type=str, default=r'../',
                        help="path to save dataset")
    parser.add_argument('--labelrate', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.53)
    parser.add_argument('--beta', type=float, default=1 / 3,
                        help='coefficient for weighted CE loss')
    parser.add_argument('--drop_method', type=str, default='dropout')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--droprate', type=float, default=0.5,
                        help='Droprate for MC-Dropout')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--norm", type=str, default='both', help="how to apply the normalizer.")

    args = parser.parse_args()
    main(args)

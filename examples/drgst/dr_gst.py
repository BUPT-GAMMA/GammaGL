import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import argparse
import copy
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from tensorlayerx.losses import mean_squared_error
from gammagl.datasets import Planetoid
from gammagl.datasets import Amazon
from gammagl.datasets import Coauthor, Flickr
from gammagl.models import GCNModel, GATModel, GraphSAGE_Full_Model, APPNPModel, GINModel, SGCModel, MixHopModel
from gammagl.transforms import DropEdge
from gammagl.utils import mask_to_index
from gammagl.data import Dataset
import random

class MyWithLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(MyWithLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self.weighted_cross_entropy(train_logits, train_y, tlx.gather(data["bald"], data["train_idx"]), data["beta"],
                                           data["num_class"], data["T"])
        return loss

    @staticmethod
    def weighted_cross_entropy(output, labels, bald, beta, num_class, T):
        bald += 1e-6
        output = tlx.softmax(output, axis=1)
        output = tlx.matmul(output, T)
        bald = bald / (tlx.reduce_mean(bald) * beta)
        labels = tlx.nn.OneHot(depth=num_class)(labels)

        tmp = tlx.reduce_sum(output * labels, axis=1)
        loss = -tlx.log(tmp)
        loss = tlx.reduce_sum(loss * bald)
        loss /= tlx.get_tensor_shape(labels)[0]
        return loss

class DR_GST():

    @staticmethod
    def test(output, y, idx_test, num_class):
        metrics = tlx.metrics.Accuracy()
        idx_test = mask_to_index(idx_test)
        test_logits = tlx.gather(tlx.identity(output), idx_test)
        test_y = tlx.gather(y, idx_test)
        acc_test = DR_GST.calculate_acc(test_logits, test_y, metrics)

        labels = tlx.gather(y, idx_test)
        loss_test = tlx.losses.softmax_cross_entropy_with_logits(tlx.gather(output, idx_test), labels)

        print(f"Test set results",
              f"loss= {tlx.convert_to_numpy(loss_test).item():.4f}",
              f"accuracy= {acc_test:.4f}")

        return acc_test, loss_test

    @staticmethod
    def generate_mask(labels):
        
        length = tlx.get_tensor_shape(labels)[0]
        train_mask = tlx.zeros(shape = (1, length), dtype = tlx.bool)
        val_mask = tlx.zeros(shape = (1, length), dtype = tlx.bool)
        test_mask = tlx.zeros(shape = (1, length), dtype = tlx.bool)

        indices = np.random.permutation(length)
        indices = tlx.convert_to_tensor(indices)

        train_ratio = 0.8
        val_ratio = 0.1
        train_size = int(train_ratio * length)
        val_size = int(val_ratio * length)

        train_mask[0][indices[:train_size]] = True
        val_mask[0][indices[train_size:train_size + val_size]] = True
        test_mask[0][indices[train_size + val_size:]] = True

        mask = (train_mask, val_mask, test_mask)

        return mask[0][0], mask[1][0], mask[2][0]

    @staticmethod
    def update_T(T, output, idx_train, labels):
        class T_loss(WithLoss):
            def __init__(self, net, loss_fn):
                super(T_loss, self).__init__(backbone=net, loss_fn=loss_fn)
            def forward(self, output, T, labels, index, n_class):
                loss = self._loss_fn(tlx.detach(tlx.gather(output, index)), 
                                     tlx.identity(tlx.gather(T, tlx.gather(labels, index)))) \
                    + self._loss_fn(tlx.identity(T), tlx.eye(n_class, n_class))

                return loss

        output = tlx.softmax(output, axis=1)
        optimizer = tlx.optimizers.Adam(lr=0.001, weight_decay=5e-4)
        x = tlx.arange(0, len(idx_train), dtype=tlx.int64)
        y = -1 * tlx.ones_like(x)
        result = tlx.where(idx_train, x, y)
        index = result[result != -1]
        n_class = tlx.convert_to_numpy(tlx.reduce_max(labels)) + 1
        loss_func = T_loss(None, mean_squared_error)
        train_one_step = TrainOneStep(loss_func, optimizer=optimizer, train_weights = [T])
        for _ in range(200):
            train_one_step(output, T, labels, index, n_class.item())

        return T

    @staticmethod
    def calculate_acc(logits, y, metrics):
        metrics.update(logits, y)
        rst = metrics.result()
        metrics.reset()
        return rst

    @staticmethod
    def generate_trainmask(dataset, labels, labelrate):
        data = dataset[0]
        train_mask = data.train_mask
        nclass = dataset.num_classes
        if not tlx.is_tensor(train_mask) == False:
            train_mask = tlx.convert_to_tensor(train_mask)
        train_index = mask_to_index(train_mask)
        train_mask = tlx.identity(train_mask)
        train_mask[:] = False
        label = labels[train_index]
        for i in range(nclass):
            class_index = mask_to_index(label == i).tolist()
            class_index = random.sample(class_index, labelrate)
            train_mask[train_index[class_index]] = True
        return train_mask

    @staticmethod
    def train(dataset, args, bald, T, idx_val, idx_train, pseudo_labels):

        net = DR_GST.get_model(args, dataset)
        data = dataset[0]
        best, bad_counter = 100, 0

        edge_index = data.edge_index
        train_idx = mask_to_index(idx_train)

        test_mask = data.test_mask
        if not tlx.is_tensor(test_mask):
            test_mask = tlx.convert_to_tensor(test_mask)
        test_idx = mask_to_index(test_mask)

        val_mask = data.val_mask
        if not tlx.is_tensor(val_mask) == False:
            val_mask = tlx.convert_to_tensor(val_mask)

        val_idx = mask_to_index(val_mask)

        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
        train_weights = net.trainable_weights

        net_with_loss = MyWithLoss(net, None)
        train_one_step = TrainOneStep(net_with_loss, optimizer, train_weights)
        metrics = tlx.metrics.Accuracy()

        train_data = {
            "x": data.x,
            "y": pseudo_labels,
            "edge_index": edge_index,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "val_idx": val_idx,
            "num_nodes": data.num_nodes,
            "num_class": dataset.num_classes,
            "bald": bald,
            "T": T,
            "beta": args.beta
        }

        for epoch in range(args.epochs):
            net.set_train()
            train_loss = train_one_step(train_data, tlx.detach(data.y))

            net.set_eval()
            logits = net(train_data['x'], train_data['edge_index'], None, train_data['num_nodes'])
            val_logits = tlx.gather(tlx.identity(logits), train_data['val_idx'])
            val_y = tlx.gather(train_data['y'], train_data['val_idx'])
            acc_val = DR_GST.calculate_acc(val_logits, val_y, metrics)

            loss_val = tlx.losses.softmax_cross_entropy_with_logits(tlx.gather(logits, idx_val), tlx.gather(pseudo_labels, idx_val))

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  val loss: {:.4f}".format(tlx.convert_to_numpy(loss_val).item()) \
                  + "  val acc: {:.4f}".format(acc_val))

            if loss_val < best:
                tlx.files.save_npz(net.all_weights, name=args.best_model_path + net.name + ".npz")
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
        unlabeled_index = mask_to_index(unlabeled_index == True)
        confidence, pred_label = DR_GST.get_confidence(output, sign)

        x = tlx.arange(0, len(confidence), dtype=tlx.int64)
        y = -1 * tlx.ones_like(x)
        result = tlx.where(confidence > threshold, x, y)
        index = result[result != -1]

        pseudo_index = []
        indices = []
        pseudo_labels, idx_train_ag = tlx.detach(labels), tlx.detach(idx_train)
        for i in index:
            if not idx_train[i]:
                indices.append(int(i))
                if i in unlabeled_index:
                    pseudo_index.append(int(i))
        update = tlx.gather(pred_label, indices)
        pseudo_labels = tlx.ops.scatter_update(pseudo_labels, tlx.convert_to_tensor(indices, dtype=tlx.int64), update)
        pseudo_index = tlx.convert_to_tensor(pseudo_index, dtype=tlx.int64)
        update = tlx.ones_like(pseudo_index, dtype=tlx.bool)
        idx_train_ag = tlx.ops.scatter_update(idx_train_ag, pseudo_index, update)
        
        idx_pseudo = tlx.zeros_like(idx_train)
        if tlx.get_tensor_shape(pseudo_index)[0] != 0:
            update = tlx.ones_like(pseudo_index, dtype=tlx.bool)
            idx_pseudo = tlx.ops.scatter_update(idx_pseudo, pseudo_index, update)

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
        for _ in range(f_pass):
            output = net(droped_data.x, droped_data.edge_index, None, droped_data.num_nodes)
            output = tlx.softmax(output, axis=1)
            out_list.append(output)
        out_list = tlx.stack(out_list)
        out_mean = tlx.reduce_mean(out_list, axis=0)
        entropy = tlx.reduce_sum(tlx.reduce_mean(out_list * tlx.log(out_list), axis=0), axis=1)
        Eentropy = tlx.reduce_sum(out_mean * tlx.log(out_mean), axis=1)
        bald = entropy - Eentropy
        return tlx.detach(bald)

    @staticmethod
    def uncertainty_dropout(model_path, args, dataset):
        f_pass = 100
        net = DR_GST.get_model(args, dataset)
        tlx.files.load_and_assign_npz(name=model_path + net.name + ".npz", network=net)
        out_list = []
        data = dataset[0]
        # with torch.no_grad():
        for _ in range(f_pass):
            output = net(data.x, data.edge_index, None, data.num_nodes)
            output = tlx.softmax(output, axis=1)
            out_list.append(output)
        out_list = tlx.stack(out_list)
        out_mean = tlx.reduce_mean(out_list, axis=0)
        entropy = tlx.reduce_sum(tlx.reduce_mean(out_list * tlx.log(out_list), axis=0), axis=1)
        Eentropy = tlx.reduce_sum(out_mean * tlx.log(out_mean), axis=1)
        bald = entropy - Eentropy
        return tlx.detach(bald)

    @staticmethod
    def load_dataset(dataset_path, name):
        if name == 'Cora':
            dataset = Planetoid(dataset_path, name)
        elif name == 'CiteSeer':
            dataset = Planetoid(dataset_path, name)
        elif name == 'PubMed':
            dataset = Planetoid(dataset_path, name)
        elif name == 'CS':
            dataset = Coauthor(dataset_path, "CS")
        elif name == 'Physics':
            dataset = Coauthor(dataset_path, "Physics")
        elif name == 'Computers':
            dataset = Amazon(dataset_path, "Computers")
        elif name == 'Photo':
            dataset = Amazon(dataset_path, "Photo")
        elif name == 'CoraFull':
            dataset = Planetoid(dataset_path, 'Cora', split='full')
        elif name == 'Flickr':
            dataset = Flickr(dataset_path, name)
        else:
            dataset = Dataset(root=dataset_path)
        return dataset

    @staticmethod
    def get_model(args, dataset, sign=True):
        model_name = args.model
        if sign:
            droprate = args.dropout
        else:
            droprate = args.droprate
        data = dataset[0]
        if model_name == 'GCN':
            model = GCNModel(feature_dim=data.num_node_features,
                             hidden_dim=args.hidden_dim,
                             num_class=dataset.num_classes,
                             drop_rate=args.droprate,
                             num_layers=args.num_layers,
                             norm=args.norm,
                             name="GCN")
        elif model_name == 'GAT':
            model = GATModel(feature_dim=data.num_node_features,
                             hidden_dim=args.hidden_dim,
                             num_class=dataset.num_classes,
                             heads=([args.nb_heads] * 1) + [args.nb_out_heads],
                             drop_rate=args.droprate,
                             num_layers=args.num_layers,
                             name="GAT")
        elif model_name == 'GraphSAGE':
            model = GraphSAGE_Full_Model(in_feats=data.num_node_features,
                                         n_hidden=args.hidden,
                                         n_classes=dataset.num_classes,
                                         n_layers=args.num_layers,
                                         activation=tlx.nn.ReLU(),
                                         dropout=droprate,
                                         aggregator_type='gcn')
        elif model_name == 'APPNP':
            model = APPNPModel(feature_dim=data.num_node_features,
                               num_class=dataset.num_classes,
                               iter_K=10,
                               alpha=0.1,
                               drop_rate=droprate,
                               name="APPNP")
        elif model_name == 'GIN':
            model = GINModel(in_channels=data.num_node_features,
                             hidden_channels=args.hidden,
                             out_channels=dataset.num_classes,
                             num_layers=args.num_layers)
        elif model_name == 'SGC':
            model = SGCModel(feature_dim=data.num_node_features,
                             num_class=dataset.num_classes,
                             iter_K=2)
        elif model_name == 'MixHop':
            model = MixHopModel(feature_dim=data.num_node_features,
                                hidden_dim=args.hidden,
                                out_dim=dataset.num_classes,
                                p=[0, 1, 2],
                                drop_rate=droprate,
                                num_layers=args.num_layers,
                                norm=True,
                                name="MixHop")

        return model


def main(args):
    dataset_path = args.dataset_path
    dataset_name = args.dataset
    dataset = DR_GST.load_dataset(dataset_path, dataset_name)

    data = dataset[0]
    labels = data.y
    num_node = data.num_nodes
    num_class = dataset.num_classes

    if dataset_name in {'Cora', 'CiteSeer', 'PubMed', 'CoraFull'}:
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    else:
        idx_train, idx_val, idx_test = DR_GST.generate_mask(labels=labels)

    if args.drop_method == 'dropedge':
        drop_edge_data = DropEdge(p=0.1)(data)

    idx_train_ag = mask_to_index(idx_train)
    idx_val_ag = mask_to_index(idx_val)
    pseudo_labels = copy.deepcopy(labels)
    bald = tlx.ones((num_node,))
    T = tlx.nn.Parameter(data = tlx.eye(num_class, num_class))

    for s in range(args.stage):
        best_output = DR_GST.train(dataset, args, bald, T, idx_val_ag, idx_train_ag, pseudo_labels)
        T = DR_GST.update_T(T, best_output, idx_train, data.y)
        idx_unlabeled = ~(idx_train | idx_test | idx_val)

        if args.drop_method == 'dropout':
            bald = DR_GST.uncertainty_dropout(args.best_model_path, args, dataset)
        elif args.drop_method == 'dropedge':
            bald = DR_GST.uncertainty_dropedge(drop_edge_data, args.best_model_path, args, dataset)

        net = DR_GST.get_model(args, dataset)
        tlx.files.load_and_assign_npz(name=args.best_model_path + net.name + ".npz", network=net)
        if tlx.BACKEND == 'torch':
            net.to(data['x'].device)
        net.set_eval()

        best_output = net(data.x, data.edge_index, None, data.num_nodes)
        idx_train_ag, pseudo_labels, idx_pseudo = DR_GST.regenerate_pseudo_label(best_output, labels, idx_train,
                                                                                 idx_unlabeled, args.threshold)
        acc_test, loss_test = DR_GST.test(best_output, data.y, idx_test, num_class)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='dataset for training')
    parser.add_argument("--dataset_path", type=str, default=r'./',
                        help="path to save dataset")
    parser.add_argument('--labelrate', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--stage', type=int, default=2)
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
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    main(args)

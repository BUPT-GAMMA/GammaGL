import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
import argparse
import copy
import numpy as np
# tensorlayerx\optimizers\torch_optimizers.py 264行 loss.bakcward() -> loss.backward(retain_graph=True)
import tensorlayerx as tlx
import torch
from tensorlayerx.model import TrainOneStep, WithLoss
from tensorlayerx.losses import mean_squared_error
from gammagl.datasets import Planetoid
from gammagl.datasets import Amazon
from gammagl.datasets import Coauthor, Flickr
from gammagl.models import GCNModel, GATModel, GraphSAGE_Full_Model, APPNPModel, GINModel, SGCModel, MIXHOPModel
from gammagl.transforms import DropEdge
from gammagl.utils import add_self_loops, mask_to_index
from gammagl.data import Dataset
import random



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

    @staticmethod
    def weighted_cross_entropy(output, labels, bald, beta, num_class, T):
        bald += 1e-6
        output = tlx.softmax(output, axis=1)
        output = tlx.matmul(output, T)
        bald = bald / (tlx.reduce_mean(bald) * beta)
        labels = MyWithLoss.one_hot(labels, num_class)
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
    def test(output, y, idx_test, num_class):
        metrics = tlx.metrics.Accuracy()
        test_logits = tlx.gather(output.clone(), idx_test)
        test_y = tlx.gather(y, idx_test)
        acc_test = DR_GST.calculate_acc(test_logits, test_y, metrics)

        one_hot_labels = MyWithLoss.one_hot(y[idx_test], num_class)
        loss_test = tlx.losses.softmax_cross_entropy_with_logits(output[idx_test], one_hot_labels)

        print(f"Test set results",
              f"loss= {loss_test.item():.4f}",
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
        # indices = torch.randperm(datalength)

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
        
        if tlx.BACKEND == 'torch':
            output = torch.softmax(output, dim=1)
            T.requires_grad = True
            optimizer = torch.optim.Adam([T], lr=0.001, weight_decay=5e-4)
            mse_criterion = torch.nn.MSELoss()
            index = torch.where(idx_train)[0]
            nclass = labels.max().item() + 1
            for epoch in range(200):
                optimizer.zero_grad()
                loss = mse_criterion(output[index].detach(), T[labels[index]].clone()) + mse_criterion(T.clone(),torch.eye(nclass))
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
        if not tlx.is_tensor(train_mask) == False:
            train_mask = tlx.convert_to_tensor(train_mask)
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

        # edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes, n_loops=args.self_loops)
        edge_index = data.edge_index
        train_idx = idx_train

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

        net_with_loss = MyWithLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
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
            "num_class": graph.num_classes,
            "bald": bald,
            "T": T,
            "beta": args.beta
        }

        for epoch in range(args.epochs):
            net.set_train()
            train_loss = train_one_step(train_data, data.y.detach())

            net.set_eval()
            logits = net(train_data['x'], train_data['edge_index'], None, train_data['num_nodes'])
            val_logits = tlx.gather(logits.clone(), train_data['val_idx'])
            val_y = tlx.gather(train_data['y'], train_data['val_idx'])
            acc_val = DR_GST.calculate_acc(val_logits, val_y, metrics)


            one_hot_labels = MyWithLoss.one_hot(pseudo_labels[idx_val], train_data['num_class'])
            loss_val = tlx.losses.softmax_cross_entropy_with_logits(logits[idx_val], one_hot_labels)

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  val loss: {:.4f}".format(loss_val) \
                  + "  val acc: {:.4f}".format(acc_val))

            # model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
            #     args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)

            if loss_val < best:
                print("-------------------------")
                print(args.best_model_path + net.name + ".npz")
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
        index = mask_to_index(confidence > threshold)
        pseudo_index = []
        pseudo_labels, idx_train_ag = labels.detach(), idx_train.detach()
        for i in index:
            if i not in idx_train:
                pseudo_labels[i] = pred_label[i]
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
        for _ in range(f_pass):
            output = net(droped_data.x, droped_data.edge_index, None, droped_data.num_nodes)
            output = tlx.softmax(output, axis=1)
            out_list.append(output)
        out_list = tlx.stack(out_list)
        out_mean = tlx.reduce_mean(out_list, axis=0)
        entropy = tlx.reduce_sum(tlx.reduce_mean(out_list * tlx.log(out_list), axis=0), axis=1)
        Eentropy = tlx.reduce_sum(out_mean * tlx.log(out_mean), axis=1)
        bald = entropy - Eentropy
        return bald.detach()

    @staticmethod
    def uncertainty_dropout(model_path, args, dataset):
        f_pass = 100
        net = DR_GST.get_model(args, dataset)
        tlx.files.load_and_assign_npz(name=model_path + net.name + ".npz", network=net)
        out_list = []
        data = dataset.data
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
        return bald.detach()


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
        data = dataset.data
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
            model = MIXHOPModel(feature_dim=data.num_node_features,
                                hidden_dim=args.hidden,
                                out_dim=dataset.num_classes,
                                p=[0, 1, 2],
                                drop_rate=droprate,
                                num_layers=args.num_layers,
                                norm=True,
                                name="MixHop")

        return model


def main(args):
    # model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
    #     args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)

    dataset_path = args.dataset_path
    dataset_name = args.dataset
    dataset = DR_GST.load_dataset(dataset_path, dataset_name)

    data = dataset.data
    labels = data.y
    num_node = data.num_nodes
    num_class = dataset.num_classes

    # seed = np.random.randint(1000)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    if dataset_name in {'Cora', 'CiteSeer', 'PubMed', 'CoraFull'}:
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    else:
        idx_train, idx_val, idx_test = DR_GST.generate_mask(labels=labels)

    if args.drop_method == 'dropedge':
        drop_edge_data = DropEdge(p=0.1)(data)

    idx_train = DR_GST.generate_trainmask(dataset, labels, args.labelrate)

    idx_train_ag = mask_to_index(idx_train)
    idx_val_ag = mask_to_index(idx_val)
    pseudo_labels = copy.deepcopy(labels)
    bald = tlx.ones((num_node,))
    T = torch.nn.Parameter(torch.eye(num_class, num_class))
    T.requires_grad = False

    for s in range(args.stage):
        best_output = DR_GST.train(dataset, args, bald, T, idx_val_ag, idx_train_ag, pseudo_labels)
        T = DR_GST.update_T(T.detach(), best_output, idx_train, data.y.data)
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
    parser.add_argument("--dataset_path", type=str, default=r'../',
                        help="path to save dataset")
    parser.add_argument('--labelrate', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=2000,
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

import os
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.utils import add_self_loops, mask_to_index, calc_gcn_norm
import time
import scipy.sparse as sp
from gammagl.models import GCNModel, GCNIIModel, GATModel
import tensorlayerx.nn as nn
from tensorlayerx.model import WithLoss, TrainOneStep
from utils import CITModule, dense_mincut_pool, edge_index_to_csr_matrix, AssignmentMatricsMLP, F1score, dense_to_sparse, calculate_acc, get_model

log_softmax = nn.activation.LogSoftmax(dim=-1)

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn, cit_module, mlp, adj_matrix):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.cit_module = cit_module
        self.mlp = mlp
        self.adj_matrix = adj_matrix

    def forward(self, data, label):
        if isinstance(self.backbone_network, GCNModel):
            intermediate_features = self.backbone_network.conv[0](data['x'], data['edge_index'], None, data["num_nodes"])
            intermediate_features = self.backbone_network.relu(intermediate_features)
            logits= log_softmax(self.backbone_network(data['x'], data['edge_index'], None, data["num_nodes"]))
        elif isinstance(self.backbone_network, GATModel):
            intermediate_features = log_softmax(self.backbone_network(data['x'], data['edge_index'], data["num_nodes"]))
            logits= log_softmax(self.backbone_network(data['x'], data['edge_index'], data["num_nodes"]))
        elif isinstance(self.backbone_network, GCNIIModel):
            intermediate_features = self.backbone_network(data['x'], data['edge_index'], data["edge_weight"], data["num_nodes"])
            logits= log_softmax(self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data["num_nodes"]))

        if not isinstance(self.backbone_network, GCNModel):
            intermediate_features = nn.Linear(in_features=intermediate_features.shape[1], out_features=args.hidden_dim, name='linear_1', act=tlx.relu)(intermediate_features)
        
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        
        #CIT
        assignment_matrics, _ = self.cit_module.forward(intermediate_features, self.mlp)
        pooled_x, pooled_adj, mc_loss, o_loss = dense_mincut_pool(data['x'], self.adj_matrix, assignment_matrics)
        loss = 0.55 * self._loss_fn(train_logits, train_y) + 0.25 * mc_loss + 0.2 * o_loss
        # loss = self._loss_fn(train_logits, train_y)
        return loss

def test(net, data, args, metrics):
    net.load_weights(args.best_model_path+net.name+".npz", format='npz_dict')
    net.set_eval()

    adj_add = sp.load_npz(f"./datasets/{args.dataset}_add_{args.ss}.npz")
    adj_add = dense_to_sparse(adj_add)
    
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(adj_add, data["num_nodes"]))
    
    if isinstance(net, GATModel):
        logits= log_softmax(net(data['x'], adj_add, data["num_nodes"]))
    elif isinstance(net, GCNModel):
        logits= log_softmax(net(data['x'], adj_add, None, data["num_nodes"]))
    elif isinstance(net, GCNIIModel):
        logits = log_softmax(net(data['x'], adj_add, edge_weight, data['num_nodes']))
    
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    
    acc_test = calculate_acc(test_logits, test_y, metrics)
    f1 = F1score(test_logits, test_y)

    print('Test set results:',
          'acc_test: {:.4f}'.format(acc_test.item()),
          'f1_score: {:.4f}'.format(f1.item()))    

def main(args):
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError("Unkown dataset: {}".format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    net = get_model(args, dataset)
    
    loss = tlx.losses.softmax_cross_entropy_with_logits
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    cit_module =CITModule(clusters=args.clusters, p=args.p)
    mlp = AssignmentMatricsMLP(input_dim=args.hidden_dim, num_clusters=args.clusters, activation='relu')
    adj_matrix = edge_index_to_csr_matrix(edge_index, graph.num_nodes)
    adj_matrix = tlx.convert_to_tensor(adj_matrix.toarray(), dtype=tlx.float32)
    
    loss_func = SemiSpvzLoss(net, loss, cit_module, mlp, adj_matrix)
    # loss_func = SemiSpvzLoss(net, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.epochtimes):
        t = time.time()
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        
        if isinstance(net, GATModel):
            logits= log_softmax(net(data['x'], data['edge_index'], data["num_nodes"]))
        elif isinstance(net, GCNIIModel):
            logits= log_softmax(net(data['x'], data['edge_index'], data["edge_weight"], data["num_nodes"]))
        elif isinstance(net, GCNModel):
            logits= log_softmax(net(data['x'], data['edge_index'], None, data["num_nodes"]))
        
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        acc_val = calculate_acc(val_logits, val_y, metrics)
        
        print('Epoch: {:04d}'.format(epoch+1),
              'train_loss: {:.4f}'.format(train_loss.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            net.save_weights(args.best_model_path+net.name+".npz", format='npz_dict')
            
    test(net, data, args, metrics)


if __name__ == '__main__':
    #set argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", type=str, default="gcn")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--dataset", type=str, default="cora", help="dataset")
    parser.add_argument("--droprate", type=float, default=0.4)
    parser.add_argument("--p", type=float, default="0.2")
    parser.add_argument("--epochtimes", type=int, default=400)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dataset_path", type=str, default=r'')
    parser.add_argument("--ss", type=float, default=0.5, help="structure shift")
    parser.add_argument("--l2_coef", type=float, default=5e-4)
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
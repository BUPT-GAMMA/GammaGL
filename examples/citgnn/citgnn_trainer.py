import os
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.utils import add_self_loops, mask_to_index
import torch
import time
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from gammagl.models import GCNModel, GCNIIModel, GATModel, APPNPModel
import torch.nn as nn
from utils import CITModule, dense_mincut_pool, edge_index_to_csr_matrix, AssignmentMatricsMLP, reassign_masks, F1score, dense_to_sparse, accuracy, get_model


def test(net, data, args):
    net.set_eval()
    
    dataset = args.dataset
    ss = args.ss
    
    adj_add = sp.load_npz(f"/home/twr/.ggl/datasets/{dataset}_add_{ss}.npz")
    adj_add = dense_to_sparse(adj_add)
    
    adj_add = adj_add.to("cpu")
    
    if isinstance(net, GATModel):
        logits= F.log_softmax(net(data['x'], adj_add, data["num_nodes"]), dim=-1)
    elif isinstance(net, GCNModel) or isinstance(net, GCNIIModel):
        logits= F.log_softmax(net(data['x'], adj_add, None, data["num_nodes"]), dim=-1)
    
    with torch.no_grad():
        test_logits = tlx.gather(logits, data['test_idx'])
        test_y = tlx.gather(data['y'], data['test_idx'])
        loss_test = F.nll_loss(test_logits, test_y)
        acc_test = accuracy(test_logits, test_y)
        f1 = F1score(test_logits, test_y)
        
        print('Test set results:',
              'loss_test: {:.4f}'.format(loss_test.item()),
              'acc_test: {:.4f}'.format(acc_test.item()),
              'f1_score: {:.4f}'.format(f1.item()))

def main(args):
    
    #load dataset
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError("Unkown dataset: {}".format(args.dataset))
    
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index)
    
    train_mask, val_mask, test_mask = reassign_masks(graph, train_ratio=0.25, val_ratio=0.35, test_ratio=0.4)
    
    train_idx = mask_to_index(train_mask)
    test_idx = mask_to_index(test_mask)
    val_idx = mask_to_index(val_mask)
    
    print("Train index size:", len(train_idx))
    print("Val index size:", len(test_idx))
    print("Test index size:", len(val_idx))
    
    net = get_model(args, dataset)
    
    
    cit_module =CITModule(clusters=args.clusters, p=args.p)
    mlp = AssignmentMatricsMLP(input_dim=args.hidden_dim, num_clusters=args.clusters, activation='relu')
    
    optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=5e-4)
    adj_matrix = edge_index_to_csr_matrix(edge_index, graph.num_nodes)
    adj_matrix = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)
    
    
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }
    
    best_val_acc = 0
    for epoch in range(args.epochtimes):
        t = time.time()
        
        net.set_train()
        optimizer.zero_grad()
        
        if isinstance(net, GCNModel):
            intermediate_features = net.conv[0](data['x'], data['edge_index'], None, data["num_nodes"])
            intermediate_features = net.relu(intermediate_features)
            logits= F.log_softmax(net(data['x'], data['edge_index'], None, data["num_nodes"]))
        elif isinstance(net, GATModel):
            intermediate_features = net(data['x'], data['edge_index'], data["num_nodes"])
            logits= F.log_softmax(net(data['x'], data['edge_index'], data["num_nodes"]))
        elif isinstance(net, GCNIIModel):
            intermediate_features = net(data['x'], data['edge_index'], None, data["num_nodes"])
            logits= F.log_softmax(net(data['x'], data['edge_index'], None, data["num_nodes"]))

        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        
        if not isinstance(net, GCNModel):
            intermediate_features = nn.Linear(intermediate_features.shape[1], args.hidden_dim)(intermediate_features)
            
        
        #CIT
        assignment_matrics, _ = cit_module.forward(intermediate_features, mlp)
        
        pooled_x, pooled_adj, mc_loss, o_loss = dense_mincut_pool(data['x'], adj_matrix, assignment_matrics)
        loss_train = 0.6 * F.nll_loss(train_logits, train_y) + mc_loss * 0.2 + o_loss * 0.2
        acc_train = accuracy(train_logits, train_y)
        
        loss_train.backward()
        optimizer.step()
        
        net.set_eval()
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        loss_val = F.nll_loss(val_logits, val_y) + mc_loss + o_loss
        acc_val = accuracy(val_logits, val_y)
            
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    test(net, data, args)
    

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
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset_path", type=str, default=r'')
    parser.add_argument("--ss", type=float, default=0.5, help="structure shift")
    
    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
        
    main(args)
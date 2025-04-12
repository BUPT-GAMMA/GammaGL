import argparse
import tensorlayerx as tlx
import numpy as np
import warnings
import os.path as osp
import random
from sklearn.metrics import f1_score
from partition import partition_patch
from gammagl.datasets import Planetoid
from gammagl.utils import get_train_val_test_split
from gammagl.models.cobformer import CoBFormer
from tensorlayerx.model import TrainOneStep, WithLoss

def eval_f1(pred, label, num_classes):
    pred = tlx.convert_to_numpy(pred)
    label = tlx.convert_to_numpy(label)
    micro = f1_score(label, pred, average='micro')
    macro = f1_score(label, pred, average='macro')
    return micro, macro

class CoLoss(WithLoss):
    def __init__(self, model, loss_fn, mask, alpha=0.8, tau=0.3):
        super(CoLoss, self).__init__(backbone=model, loss_fn=loss_fn)
        self.mask = mask
        self.alpha = alpha
        self.tau = tau

    def forward(self, data, label):
        pred1, pred2 = self.backbone_network(data['x'], data['patch'], data['edge_index'], edge_weight=data['edge_weight'], num_nodes=data['num_nodes'])
        l1 = tlx.losses.softmax_cross_entropy_with_logits(pred1[self.mask], label[self.mask])
        l2 = tlx.losses.softmax_cross_entropy_with_logits(pred2[self.mask], label[self.mask])
        
        pred1_scaled = pred1 * self.tau
        pred2_scaled = pred2 * self.tau
        
        l3 = tlx.losses.softmax_cross_entropy_with_logits(pred1_scaled[~self.mask], tlx.nn.Softmax()(pred2_scaled)[~self.mask])
        l4 = tlx.losses.softmax_cross_entropy_with_logits(pred2_scaled[~self.mask], tlx.nn.Softmax()(pred1_scaled)[~self.mask])
        
        return self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
        
def co_train(model, data, label, patch, split_index, optimizer):
    model.train()
    # optimizer.zero_grad()
    edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
    
    train_weights = model.trainable_weights
    loss_func = CoLoss(model, tlx.losses.softmax_cross_entropy_with_logits, split_index['train'], model.alpha, model.tau)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    
    train_data = {
        'x': data.x,
        'patch': patch,
        'edge_index': data.edge_index,
        'edge_weight': edge_weight,
        'num_nodes': data.num_nodes,
        'label': label,
        'train_idx': split_index['train']
    }
    # loss = optimizer(train_data)
    loss = train_one_step(train_data, label)
    model.eval()
    
    with torch.no_grad():
        pred1, pred2 = model(data.x, patch, data.edge_index, edge_weight=edge_weight, num_nodes=data.num_nodes)
        y = data.y
        if len(y.shape) > 1:
            y = y.view(-1)  
            
        num_classes = int(tlx.reduce_max(y) + 1)
        
        y1_ = tlx.argmax(pred1, axis=1)
        if len(y1_.shape) > 1:
            y1_ = y1_.view(-1)
            
        micro_val1, macro_val1 = eval_f1(y1_[split_index['valid']], y[split_index['valid']], num_classes)
        micro_test1, macro_test1 = eval_f1(y1_[split_index['test']], y[split_index['test']], num_classes)

        y2_ = tlx.argmax(pred2, axis=1)
        if len(y2_.shape) > 1:
            y2_ = y2_.view(-1)
            
        micro_val2, macro_val2 = eval_f1(y2_[split_index['valid']], y[split_index['valid']], num_classes)
        micro_test2, macro_test2 = eval_f1(y2_[split_index['test']], y[split_index['test']], num_classes)

    return micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2



def co_early_stop_train(epochs, patience, model, data, label, patch, split_index, optimizer, show_details,
                        postfix, save_path=None):
    best_epoch1 = 0
    best_epoch2 = 0 
    acc_val1_max = 0.
    acc_val2_max = 0.
    logger = []
    max_val = -10000

    for epoch in range(1, epochs + 1):
        micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2 = co_train(
            model, data, label, patch, split_index, optimizer)
        logger.append(
            [micro_val1, micro_test1, macro_val1, macro_test1, micro_val2, micro_test2, macro_val2, macro_test2])

        if show_details and epoch % 50 == 0:
            print(
                f'(T) | Epoch={epoch:03d}\n',
                f'micro_val1={micro_val1:.4f}, micro_test1={micro_test1:.4f}, macro_val1={macro_val1:.4f}, macro_test1={macro_test1:.4f}\n', 
                f'micro_val2={micro_val2:.4f}, micro_test2={micro_test2:.4f}, macro_val2={macro_val2:.4f}, macro_test2={macro_test2:.4f}\n')

    logger = tlx.convert_to_tensor(logger)
    ind = tlx.argmax(logger, axis=0)

    res_gnn = []
    res_trans = []

    res_gnn.append(logger[ind[0]][0])
    res_gnn.append(logger[ind[0]][1]) 
    res_gnn.append(logger[ind[2]][2])
    res_gnn.append(logger[ind[2]][3])
    res_gnn.append(logger[ind[1]][1])
    res_gnn.append(logger[ind[3]][3])

    res_trans.append(logger[ind[4]][4])
    res_trans.append(logger[ind[4]][5])
    res_trans.append(logger[ind[6]][6]) 
    res_trans.append(logger[ind[6]][7])
    res_trans.append(logger[ind[5]][5])
    res_trans.append(logger[ind[7]][7])

    return res_gnn, res_trans



def main(args, device, data, patch, split_idx, alpha, tau, postfix):
    if hasattr(args, 'runs'):
        runs = args.runs
    else:
        runs = 5  
        
    results = [[], []]
    for r in range(runs):

        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        gcn_wd = args.gcn_wd
        num_hidden = args.num_hidden
        activation = tlx.relu
        num_layers = args.num_layers
        n_head = args.n_head
        num_epochs = args.num_epochs
        gcn_type = args.gcn_type
        gcn_layers = args.gcn_layers
        gcn_use_bn = args.gcn_use_bn
        use_patch_attn = args.use_patch_attn
        show_details = args.show_details
        
        try:
            patch_copy = tlx.cast(patch, dtype=tlx.int64)
        except:
            patch_copy = tlx.convert_to_tensor(patch, dtype=tlx.int64)
        
        num_nodes = data.num_nodes
        num_classes = data.y.max() + 1
        num_features = data.x.shape[-1]     
        # Convert label to one-hot encoding and cast to float type
        label = tlx.nn.OneHot(num_classes)(data.y)
        label = tlx.cast(label, dtype=tlx.float32)

        model = CoBFormer(num_nodes, num_features, num_hidden, num_classes, activation, layers=num_layers,
                         gcn_layers=gcn_layers, gcn_type=gcn_type, n_head=n_head, alpha=alpha, tau=tau,
                         gcn_use_bn=gcn_use_bn, use_patch_attn=use_patch_attn)

        optimizer = tlx.optimizers.Adam(lr=learning_rate, weight_decay=weight_decay)

        patience = num_epochs

        res_gnn, res_trans = co_early_stop_train(
            num_epochs, patience,
            model, data, label,
            patch_copy, split_idx,
            optimizer, show_details,
            postfix)
            
        print(f"=== Train Final (Run {r+1}/{runs}) ===")
        print(
            f'micro_val1={res_gnn[0]:.4f}, micro_test1={res_gnn[1]:.4f}, macro_val1={res_gnn[2]:.4f}, macro_test1={res_gnn[3]:.4f}, micro_best1={res_gnn[4]:.4f}, macro_best1={res_gnn[5]:.4f},\n',
            f'micro_val2={res_trans[0]:.4f}, micro_test2={res_trans[1]:.4f}, macro_val2={res_trans[2]:.4f}, macro_test2={res_trans[3]:.4f}, micro_best2={res_trans[4]:.4f}, macro_best2={res_trans[5]:.4f}\n')

        results[0].append(res_gnn)
        results[1].append(res_trans)
    
    # statistics results
    print(f"==== Final GNN====")
    result = tlx.convert_to_tensor(results[0]) * 100.  
    print(result)
    print(f"max: {tlx.ops.reduce_max(result, axis=0)}")  
    print(f"min: {tlx.ops.reduce_min(result, axis=0)}")   
    print(f"mean: {tlx.ops.reduce_mean(result, axis=0)}")  
    print(f"std: {tlx.ops.reduce_std(result, axis=0)}")  

    print(f'GNN Micro: {tlx.ops.reduce_mean(result, axis=0)[1]:.2f} ± {tlx.ops.reduce_std(result, axis=0)[1]:.2f}')
    print(f'GNN Macro: {tlx.ops.reduce_mean(result, axis=0)[3]:.2f} ± {tlx.ops.reduce_std(result, axis=0)[3]:.2f}')

    print(f"==== Final Trans====")
    result = tlx.convert_to_tensor(results[1]) * 100.
    print(result)
    print(f"max: {tlx.ops.reduce_max(result, axis=0)}")
    print(f"min: {tlx.ops.reduce_min(result, axis=0)}")
    print(f"mean: {tlx.ops.reduce_mean(result, axis=0)}")
    print(f"std: {tlx.ops.reduce_std(result, axis=0)}")

    print(f'Trans Micro: {tlx.ops.reduce_mean(result, axis=0)[1]:.2f} ± {tlx.ops.reduce_std(result, axis=0)[1]:.2f}')
    print(f'Trans Macro: {tlx.ops.reduce_mean(result, axis=0)[3]:.2f} ± {tlx.ops.reduce_std(result, axis=0)[3]:.2f}')
    
    return results

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gcn_wd', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=6)
    parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--gcn_use_bn', action='store_true', help='gcn use batch norm')
    parser.add_argument('--use_patch_attn', action='store_true', help='transformer use patch attention')
    parser.add_argument('--show_details', type=bool, default=True)
    parser.add_argument('--gcn_type', type=int, default=1)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--n_patch', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.6)
    parser.add_argument('--valid_prop', type=float, default=.2)
    parser.add_argument('--alpha', type=float, default=.8)
    parser.add_argument('--tau', type=float, default=.3)

    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    device = tlx.get_device()
    n_patch = args.n_patch
    alpha = args.alpha
    tau = args.tau
    load_path = None

    postfix = "test"
    runs = 5
    print("n_patch: ", n_patch)
    
    dataset = Planetoid(args.dataset)
    data = dataset[0]

    data.train_mask = tlx.convert_to_numpy(data.train_mask)
    data.val_mask = tlx.convert_to_numpy(data.val_mask)
    data.test_mask = tlx.convert_to_numpy(data.test_mask)
    # Pad a dimension with value 0 at the end of each mask (1D array) using np.pad(mask, (0, 1), mode='constant')
    data.train_mask = np.pad(data.train_mask, (0, 1), mode='constant')
    data.val_mask = np.pad(data.val_mask, (0, 1), mode='constant')
    data.test_mask = np.pad(data.test_mask, (0, 1), mode='constant')

    split_idx = {
        'train': data.train_mask,
        'valid': data.val_mask,
        'test': data.test_mask
    }


    patch = partition_patch(data, n_patch, load_path)
    batch_size = args.batch_size

    main(args, device, data, patch, split_idx, alpha, tau, postfix)
import argparse
import tensorlayerx as tlx
import numpy as np
import warnings
import os.path as osp
import random
from sklearn.metrics import f1_score
import torch.optim as optim
import torch
from partition import partition_patch
from gammagl.datasets import Planetoid
from gammagl.utils import get_train_val_test_split
from gammagl.models.cobformer import CoBFormer

def eval_f1(pred, label, num_classes):
    # 将tlx tensor转换为numpy数组
    pred = tlx.convert_to_numpy(pred)
    label = tlx.convert_to_numpy(label)
    micro = f1_score(label, pred, average='micro')
    macro = f1_score(label, pred, average='macro')
    return micro, macro


def co_train(model, data, label, patch, split_index, optimizer):
    model.train()
    optimizer.zero_grad()
    # 如果data中有edge_weight就使用，否则为None
    edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
    pred1, pred2 = model(data.x, patch, data.edge_index, edge_weight=edge_weight, num_nodes=data.num_nodes)
    loss = model.loss(pred1, pred2, label, split_index['train'])
    loss.backward()
    optimizer.step()
    # eval
    model.eval()
    with torch.no_grad():
        pred1, pred2 = model(data.x, patch, data.edge_index, edge_weight=edge_weight, num_nodes=data.num_nodes)

        # 直接使用data.y，如果需要确保是一维张量，可以使用索引或view操作
        y = data.y
        if len(y.shape) > 1:
            y = y.view(-1)  # 或使用 y = y[:, 0] 如果是二维的
            
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


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)  
    tlx.set_device(device=f'gpu:{args.gpu_id}', id=args.gpu_id)

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

def run(args, device, data, patch, split_idx, alpha, tau, postfix):
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
    
    # 确保patch是long类型
    try:
        patch = tlx.cast(patch, dtype=tlx.int64)
    except:
        patch = tlx.convert_to_tensor(patch, dtype=tlx.int64)
    
    num_nodes = data.num_nodes
    num_classes = data.y.max() + 1
    num_features = data.x.shape[-1]
    # 将label转换为one-hot编码，且转换为浮点类型label = F.one_hot(data.label, num_classes).float()
    label = tlx.nn.OneHot(num_classes)(data.y)
    label = tlx.cast(label, dtype=tlx.float32)

    model = CoBFormer(num_nodes, num_features, num_hidden, num_classes, activation, layers=num_layers,
                     gcn_layers=gcn_layers, gcn_type=gcn_type, n_head=n_head, alpha=alpha, tau=tau,
                     gcn_use_bn=gcn_use_bn, use_patch_attn=use_patch_attn)

    if args.dataset in ['film', 'CiteSeer', 'Cora', 'PubMed', "Deezer"]:
        # 为不同参数组设置不同的权重衰减
        # bga_params = []
        # gcn_params = []
        # for name, param in model.named_parameters():
        #     if 'bga' in name:
        #         bga_params.append(param)
        #     elif 'gcn' in name:
        #         gcn_params.append(param)
                
        optimizer = optim.Adam([
            {'params': model.bga.parameters(), 'weight_decay': weight_decay},
            {'params': model.gcn.parameters(), 'weight_decay': gcn_wd}
        ], lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    patience = num_epochs

    res_gnn, res_trans = co_early_stop_train(
        num_epochs, patience,
        model, data, label,
        patch, split_idx,
        optimizer, show_details,
        postfix)
        
    print("=== Train Final ===")
    print(
        f'micro_val1={res_gnn[0]:.4f}, micro_test1={res_gnn[1]:.4f}, macro_val1={res_gnn[2]:.4f}, macro_test1={res_gnn[3]:.4f}, micro_best1={res_gnn[4]:.4f}, macro_best1={res_gnn[5]:.4f},\n',
        f'micro_val2={res_trans[0]:.4f}, micro_test2={res_trans[1]:.4f}, macro_val2={res_trans[2]:.4f}, macro_test2={res_trans[3]:.4f}, micro_best2={res_trans[4]:.4f}, macro_best2={res_trans[5]:.4f}\n')

    return res_gnn, res_trans

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gcn_wd', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=6)
    parser.add_argument('--num_hidden', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1, help='层数')
    parser.add_argument('--n_head', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_epochs', type=int, default=500, help='训练轮数')
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

    fix_seed(args.seed)

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    n_patch = args.n_patch
    alpha = args.alpha
    tau = args.tau
    load_path = None
    if args.dataset in ['ogbn-products']:
        load_path = f'Data/partition/{args.dataset}_partition_{n_patch}.pt'

    postfix = "test"
    runs = 5
    print("n_patch: ", n_patch)
    
    dataset = Planetoid(path, args.dataset)
    data = dataset[0]

    data.train_mask = tlx.convert_to_numpy(data.train_mask)
    data.val_mask = tlx.convert_to_numpy(data.val_mask)
    data.test_mask = tlx.convert_to_numpy(data.test_mask)
    # 在各个mask后面pad一个维度，pad的值为0，mask是一维数组，用np.pad(mask, (0, 1), mode='constant')
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

    results = [[], []]
    for r in range(runs):        
        res_gnn, res_trans = run(args, device, data, patch, split_idx, alpha, tau, postfix)
        results[0].append(res_gnn)
        results[1].append(res_trans)

    print(f"==== Final GNN====")
    result = tlx.convert_to_tensor(results[0]) * 100.  # 替换torch.tensor
    print(result)
    print(f"max: {tlx.ops.reduce_max(result, axis=0)}")  # 使用 tlx.ops.reduce_max
    print(f"min: {tlx.ops.reduce_min(result, axis=0)}")   # 使用 tlx.ops.reduce_min
    print(f"mean: {tlx.ops.reduce_mean(result, axis=0)}")  # 使用 tlx.ops.reduce_mean
    print(f"std: {tlx.ops.reduce_std(result, axis=0)}")  # 使用 tlx.ops.reduce_std

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
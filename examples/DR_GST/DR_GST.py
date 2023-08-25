import copy
import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TL_BACKEND'] = 'torch'
# tensorlayerx\optimizers\torch_optimizers.py 264行 loss.bakcward() -> loss.backward(retain_graph=True)
import argparse
import tensorlayerx as tlx
from gammagl.models import DR_GST
from gammagl.utils import mask_to_index
from gammagl.transforms import DropEdge
import torch


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def main(args):
    model_path = './save_model/%s-%s-%d-%f-%f-%f-%s.pth' % (
        args.model, args.dataset, args.labelrate, args.threshold, args.beta, args.droprate, args.drop_method)

    dataset_path = args.dataset_path
    dataset_name = args.dataset
    dataset = DR_GST.load_dataset(dataset_path, dataset_name)

    data = dataset.data
    labels = data.y
    num_node = data.num_nodes
    num_class = dataset.num_classes

    seed = np.random.randint(1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if dataset_name in {'Cora', 'CiteSeer', 'PubMed', 'CoraFull'}:
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask
    else:
        idx_train, idx_val, idx_test = DR_GST.generate_mask(dataset=dataset, labels=labels, labelrate=20)

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
        best_output = DR_GST.train1(dataset, args, bald, T, idx_val_ag, idx_train_ag, pseudo_labels)
        T = DR_GST.update_T(T.detach(), best_output, idx_train, data.y.data)
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
        acc_test, loss_test = DR_GST.test(best_output, data.y, idx_test, num_class)
        idx_train_ag, pseudo_labels, idx_pseudo = DR_GST.regenerate_pseudo_label(best_output, labels, idx_train,
                                                                                 idx_unlabeled, args.threshold)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='dataset for training')
    parser.add_argument("--dataset_path", type=str, default=r'../',
                        help="path to save dataset")
    parser.add_argument('--labelrate', type=int, required=True, default=20)
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

    args = parser.parse_args()
    main(args)

import argparse

import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '



import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.model import WithLoss, TrainOneStep
import numpy as np
from load_data import mvgrl_load
from tqdm import tqdm


from gammagl.data import BatchGraph


from gammagl.models.mvgrl import MVGRL_Graph
from gammagl.utils.tu_utils import linearsvc


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data["edge_index"], data["diff_edge"], data["diff_weight"], data["feat"],
                              data["ptr"], data["batch"])
        return loss


def collate(batch):
    ''' collate function for building the graph dataloader'''
    graphs, diff_graphs, labels = map(list, zip(*batch))

    # generate batched graphs and labels


    batched_graph = BatchGraph.from_data_list(graphs)
    batched_labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)
    batched_diff_graph = BatchGraph.from_data_list(diff_graphs)

    ptr = batched_graph.ptr
    batch = np.zeros((batched_graph.x.shape[0],), dtype=np.int64)

    return batched_graph, batched_diff_graph, batched_labels


def main(args):
    dataset = mvgrl_load(args.dataset_path, args.dataset)
    print('=' * 300)
    graphs, diff_graphs, labels = map(list, zip(*dataset))
    print('Name of graphs:', args.dataset)
    print('Number of graphs:', len(graphs))

    whole_graph = BatchGraph.from_data_list(graphs)
    whole_diff = BatchGraph.from_data_list(diff_graphs)


    train_loader = DataLoader(dataset, collate_fn=collate, batch_size=args.batch_size, shuffle=True)
    model = MVGRL_Graph(in_feat=whole_graph.x.shape[1], out_feat=args.hidden_dim, num_layers=args.num_layers)
    embs = model.get_embedding(whole_graph.edge_index, whole_diff.edge_index, whole_diff.edge_weight, whole_graph.x,
                               whole_graph.ptr)
    lbls = tlx.convert_to_tensor(labels)
    acc_mean, acc_std = linearsvc(embs, lbls)
    print('accuracy_mean, {:.4f}'.format(acc_mean))
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = model.trainable_weights
    loss_func = Unsupervised_Loss(model)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    best = float('inf')
    cnt_wait = 0
    for epoch in tqdm(range(args.n_epoch)):
        loss_all = 0
        model.set_train()
        for batched_graph, batched_diff_graph, batched_labels in train_loader:
            data = {"edge_index": batched_graph.edge_index,
                    "diff_edge": batched_diff_graph.edge_index, "diff_weight": batched_diff_graph.edge_weight,
                    "feat": batched_graph.x, "ptr": batched_graph.ptr, "batch": batched_graph.batch}
            loss = train_one_step(data=data, label=tlx.convert_to_tensor([1])).item()
            loss_all += loss
        print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0

            model.save_weights(args.best_model_path + "MVGRL_"+args.dataset+".npz", format='npz_dict')

        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break
    print('Training End')

    model.load_weights(args.best_model_path + "MVGRL_"+args.dataset+".npz", format='npz_dict')
    model.set_eval()

    embs = model.get_embedding(whole_graph.edge_index, whole_diff.edge_index, whole_diff.edge_weight,
                                      whole_graph.x,
                                      whole_graph.ptr)

    acc_mean, acc_std = linearsvc(embs, lbls)
    print('accuracy_mean, {:.4f}'.format(acc_mean))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")

    parser.add_argument("--n_epoch", type=int, default=20, help="number of epoch")

    parser.add_argument("--hidden_dim", type=int, default=32, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='dataset, MUTAG, PTC_MR, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping steps.')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of GCNConv layer')
    parser.add_argument("--dataset_path", type=str, default=r'./', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()

    main(args)

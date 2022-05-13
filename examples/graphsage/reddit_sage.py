# import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
from tensorlayerx.model import WithLoss, TrainOneStep
from tqdm import tqdm

from gammagl.datasets import Reddit
import tensorlayerx as tlx
import argparse
from gammagl.loader.Neighbour_sampler import Neighbor_Sampler
from gammagl.models import GraphSAGE_Sample_Model


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'], data['subgs'])
        train_logits = logits
        train_label = label
        loss = self._loss_fn(train_logits, train_label)
        # train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_label))
        # print("train_acc:{:.4f}".format(train_acc))

        return loss


def evaluate(net, feat, loader, y, mask, metrics):
    net.set_eval()
    logits = net.inference(feat, loader)
    _logits = logits[mask]
    _label = y[mask]
    metrics.update(_logits, _label)
    acc = metrics.result()  # [0]
    metrics.reset()
    return acc


def main(args):
    # load reddit dataset
    dataset = Reddit(args.dataset_path, args.dataset)
    # dataset.process()  # suggest to execute explicitly so far
    graph = dataset.data
    train_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                    dst_nodes=tlx.arange(graph.x.shape[0])[graph.train_mask],
                                    sample_lists=[25, 10], batch_size=1024, shuffle=True, num_workers=0)

    val_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                  dst_nodes=tlx.arange(graph.x.shape[0])[graph.val_mask],
                                  sample_lists=[-1], batch_size=2048 * 2, shuffle=False, num_workers=0)
    test_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                   dst_nodes=tlx.arange(graph.x.shape[0])[graph.test_mask],
                                   sample_lists=[-1], batch_size=2048 * 2, shuffle=False, num_workers=0)

    x = tlx.convert_to_tensor(graph.x)
    # edge_index = graph.edge_index
    y = tlx.convert_to_tensor(graph.y)

    net = GraphSAGE_Sample_Model(in_feat=x.shape[1],
                                 hid_feat=args.hidden_dim,
                                 out_feat=graph.num_class,
                                 drop_rate=args.drop_rate,
                                 num_layers=args.num_layers)
    optimizer = tlx.optimizers.Adam(args.lr)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')
        for dst_node, adjs, all_node in train_loader:
            net.set_train()
            # input : sampled subgraphs, sampled node's feat
            data = {"x": tlx.gather(x, all_node),
                    "subgs": adjs}
            train_loss = train_one_step(data, tlx.gather(y, dst_node))
            pbar.update(len(dst_node))
            print("Epoch [{:0>3d}] ".format(epoch + 1) + "  train loss: {:.4f}".format(train_loss.item()))
        val_acc = evaluate(net, x, val_loader, y, graph.val_mask, metrics)
        test_acc = evaluate(net, x, test_loader, y, graph.test_mask, metrics)
        print("val acc: {:.4f} || test acc{:.4f}".format(val_acc, test_acc))



if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=20, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=256, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='reddit', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../reddit', help="path to save dataset")
    # parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()

    main(args)

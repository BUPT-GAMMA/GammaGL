import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
# set your backend here, default `tensorflow`

from gammagl.utils import mask_to_index


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

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['subgs'])
        loss = self._loss_fn(logits, tlx.gather(data['y'], data['dst_node']))
        return loss


def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """

    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def main(args):
    # load reddit dataset
    dataset = Reddit(args.dataset_path)
    # dataset.process()  # suggest to execute explicitly so far
    graph = dataset[0]
    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)


    train_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                    dst_nodes=tlx.convert_to_numpy(train_idx),
                                    sample_lists=[25, 10], batch_size=2048, shuffle=True, num_workers=0)

    val_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                  dst_nodes=tlx.convert_to_numpy(val_idx),
                                  sample_lists=[-1], batch_size=2048 * 2, shuffle=False, num_workers=0)
    test_loader = Neighbor_Sampler(edge_index=graph.edge_index.numpy(),
                                   dst_nodes=tlx.convert_to_numpy(test_idx),
                                   sample_lists=[-1], batch_size=2048 * 2, shuffle=False, num_workers=0)

    x = tlx.convert_to_tensor(graph.x)
    y = tlx.convert_to_tensor(graph.y, dtype=tlx.int64)

    net = GraphSAGE_Sample_Model(in_feat=dataset.num_node_features,
                                 hid_feat=args.hidden_dim,
                                 out_feat=dataset.num_classes,
                                 drop_rate=args.drop_rate,
                                 num_layers=args.num_layers)
    optimizer = tlx.optimizers.Adam(args.lr)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in range(args.n_epoch):
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')
        for dst_node, adjs, all_node in train_loader:
            net.set_train()
            # input : sampled subgraphs, sampled node's feat
            data = {"x": tlx.gather(x, tlx.convert_to_tensor(all_node)),
                    "y": y,
                    "dst_node": tlx.convert_to_tensor(dst_node),
                    "subgs": adjs}
            # label is not used
            train_loss = train_one_step(data, tlx.convert_to_tensor([0]))
            pbar.update(len(dst_node))
            print("Epoch [{:0>3d}] ".format(epoch + 1) + "  train loss: {:.4f}".format(train_loss.item()))

        logits = net.inference(x, val_loader, data['x'])
        if tlx.BACKEND == 'torch':
            val_idx = val_idx.to(data['x'].device)
        val_logits = tlx.gather(logits, val_idx)
        val_y = tlx.gather(data['y'], val_idx)
        val_acc = calculate_acc(val_logits, val_y, metrics)

        logits = net.inference(x, test_loader, data['x'])
        test_logits = tlx.gather(logits, test_idx)
        test_y = tlx.gather(data['y'], test_idx)
        test_acc = calculate_acc(test_logits, test_y, metrics)

        print("val acc: {:.4f} || test acc{:.4f}".format(val_acc, test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=256, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.8, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='reddit', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../reddit', help="path to save dataset")
    # parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    args = parser.parse_args()

    main(args)

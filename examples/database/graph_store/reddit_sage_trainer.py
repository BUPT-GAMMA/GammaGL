import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from gammagl.utils import mask_to_index
from tensorlayerx.model import WithLoss, TrainOneStep
from tqdm import tqdm
import tensorlayerx as tlx
import argparse
from gammagl.loader.neighbor_sampler import NeighborSampler
from gammagl.models import GraphSAGE_Sample_Model
import numpy as np
from gdbi.ggl import Neo4jFeatureStore, Neo4jGraphStore

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
    uri = 'bolt://localhost:7687'
    user_name= 'neo4j'
    password= 'neo4j'

    feature_store = Neo4jFeatureStore(uri=uri, user_name=user_name, password=password)
    graph_store = Neo4jGraphStore(uri=uri, user_name=user_name, password=password)
    
    edge_index = graph_store['reddit_edge', 'coo']
    
    with np.load('idx.npz') as data:
        train_idx = tlx.convert_to_tensor(data['train_idx'])
        val_idx = tlx.convert_to_tensor(data['val_idx'])
        test_idx = tlx.convert_to_tensor(data['test_idx'])
    train_loader = NeighborSampler(edge_index=edge_index,
                                   node_idx=train_idx,
                                   sample_lists=[25, 10], batch_size=2048, shuffle=True, num_workers=0)

    val_loader = NeighborSampler(edge_index=edge_index,
                                 node_idx=val_idx,
                                 sample_lists=[-1], batch_size=2048 * 2, shuffle=False, num_workers=0)
    test_loader = NeighborSampler(edge_index=edge_index,
                                  node_idx=test_idx,
                                  sample_lists=[-1], batch_size=2048 * 2, shuffle=False, num_workers=0)


    net = GraphSAGE_Sample_Model(in_feat=602,
                                 hid_feat=args.hidden_dim,
                                 out_feat=41,
                                 drop_rate=args.drop_rate,
                                 num_layers=args.num_layers)
    optimizer = tlx.optimizers.Adam(args.lr)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    y = feature_store['reddit_node', 'y', tlx.arange(start=0, limit=232965)]
    y = tlx.reshape(tlx.cast(y, dtype=tlx.int64), (-1,))
    
    for epoch in range(args.n_epoch):
        pbar = tqdm(total=int(len(train_loader.dataset)))
        pbar.set_description(f'Epoch {epoch:02d}')
        for dst_node, n_id, adjs in train_loader:
            net.set_train()
            # input : sampled subgraphs, sampled node's feat
            data = {"x": feature_store['reddit_node', 'x', n_id],
                    "y": y,
                    "dst_node": dst_node,
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
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    # parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--gpu", type=int, default=-1)

    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
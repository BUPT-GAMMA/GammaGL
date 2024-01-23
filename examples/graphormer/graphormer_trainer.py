import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import sys
import tensorlayerx as tlx
from gammagl.datasets import MoleculeNet
from gammagl.loader import DataLoader
from gammagl.models import Graphormer
from tensorlayerx.dataflow import Subset
from sklearn.model_selection import train_test_split
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.layers.pool import global_mean_pool
from tqdm import tqdm

class Loss(WithLoss):
    def __init__(self, net, loss_fn):
        super(Loss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data)
        output = global_mean_pool(logits, data.batch)
        loss = self._loss_fn(output, label)
        return loss

def main(args):
    if str.lower(args.dataset) not in ['esol']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = MoleculeNet(root=args.dataset_path, name=args.dataset)
    test_ids, train_ids = train_test_split([i for i in range(len(dataset))], test_size=args.test_size, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(dataset, test_ids), batch_size=args.batch_size)

    model = Graphormer(
        num_layers=args.num_layers,
        input_node_dim=dataset.num_node_features,
        node_dim=args.node_dim,
        input_edge_dim=dataset.num_edge_features,
        edge_dim=args.edge_dim,
        output_dim=tlx.get_tensor_shape(dataset[0].y)[1],
        n_heads=args.heads,
        max_in_degree=args.max_in_degree,
        max_out_degree=args.max_out_degree,
        max_path_distance=args.max_path_distance,
        name='Graphormer'
    )

    optimizer = tlx.optimizers.Adam(lr=args.lr)
    train_weights = model.trainable_weights
    loss_func = Loss(net=model, loss_fn=tlx.losses.absolute_difference_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    loss = sys.float_info.max
    for epoch in range(args.n_epoch):
        model.set_train()
        batch_loss = 0.0
        for batch in tqdm(train_loader):
            batch_loss += train_one_step(batch, batch.y).item()

        print((f'TRAIN_LOSS: {batch_loss / len(train_ids):.4f}'))
        
        if batch_loss < loss:
            loss = batch_loss
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    model.set_eval()
    batch_loss = 0.0
    for batch in tqdm(test_loader):
        loss = loss_func(batch, batch.y)
        batch_loss += loss.item()

    print((f'EVAL_LOSS: {batch_loss / len(test_ids):.4f}'))

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of epoch")
    parser.add_argument("--node_dim", type=int, default=128, help="dimention of node embedding")
    parser.add_argument("--edge_dim", type=int, default=128, help="dimention of edge embedding")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers")
    parser.add_argument("--test_size", type=float, default=0.8, help="the scale of test set")
    parser.add_argument('--dataset', type=str, default='esol', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch size")
    parser.add_argument("--heads", type=int, default=4, help="number of heads for stablization")
    parser.add_argument("--max_in_degree", type=int, default=5, help="max in degree of node")
    parser.add_argument("--max_out_degree", type=int, default=5, help="max out degree of node")
    parser.add_argument("--max_path_distance", type=int, default=5, help="max path distance")

    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)

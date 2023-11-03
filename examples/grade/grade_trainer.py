import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
import argparse
from gammagl.models.grade import GRADE
from aug import degree_aug
from utils import load, linear_clf
from tensorlayerx.nn.layers.activation import ReLU, PRelu
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils.corrupt_graph import add_self_loops, dfde_norm_g
from warnings import filterwarnings
filterwarnings("ignore")

class Unsupervised_Loss(WithLoss):
    def __init__(self, model):
        super(Unsupervised_Loss, self).__init__(backbone=model, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data['feat1'], data['edge1'], data['feat2'], data['edge2'])
        return loss

def main(args):
    tlx.set_device("GPU", args.gpu_id)
    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({'relu': ReLU(), 'prelu': PRelu()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    edge_index, feat, labels, train_mask, test_mask, degree, num_nodes = load(args.dataset, args.mode, args.dataset_path)
    in_dim = tlx.get_tensor_shape(feat)[1]

    model = GRADE(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    optimizer = tlx.optimizers.Adam(lr=lr, weight_decay=wd)
    train_weights = model.trainable_weights
    loss_func = Unsupervised_Loss(model)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    model.set_train()
    for epoch in range(epochs):
        #warm up
        if epoch<args.warmup:
            graph1 = dfde_norm_g(edge_index, feat, drop_feature_rate_1,
                                 drop_edge_rate_1)
            graph2 = dfde_norm_g(edge_index, feat, drop_feature_rate_2,
                                 drop_edge_rate_2)
        else:
            added_loop_edge_index, _ = add_self_loops(edge_index)
            embeds = model.get_embedding(feat,added_loop_edge_index)
            graph1, graph2 = degree_aug(edge_index, feat, embeds, degree,
                                        drop_feature_rate_1, drop_edge_rate_1,
                                        drop_feature_rate_2, drop_edge_rate_2,
                                        args.threshold, num_nodes)
        data = {"feat1":graph1.x, "edge1": graph1.edge_index,
                "feat2":graph2.x, "edge2": graph2.edge_index}
        loss = train_one_step(data, label=None)
        print(f'Epoch={epoch+1}, loss={loss.item():.4f}')
    print("=== Final ===")
    model.set_eval()
    added_loop_edge_index, _ = add_self_loops(edge_index)
    embeds = model.get_embedding(feat, added_loop_edge_index)
    '''Evaluation Embeddings'''
    linear_clf(embeds, labels, tlx.convert_to_numpy(train_mask), tlx.convert_to_numpy(test_mask), degree, args.dataset)


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--warmup', type=int, default=200, help='Warmup of training.')
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--threshold', type=int, default=9, help='Definition of low-degree nodes.')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs.')
    parser.add_argument("--hid_dim", type=int, default=256, help='Hidden layer dim.')
    parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')
    parser.add_argument('--der1', type=float, default=0.20, help='Drop edge ratio of the 1st augmentation.')
    parser.add_argument('--der2', type=float, default=0.20, help='Drop edge ratio of the 2nd augmentation.')
    parser.add_argument('--dfr1', type=float, default=0.20, help='Drop feature ratio of the 1st augmentation.')
    parser.add_argument('--dfr2', type=float, default=0.20, help='Drop feature ratio of the 2nd augmentation.')
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    args = parser.parse_args()
    main(args)

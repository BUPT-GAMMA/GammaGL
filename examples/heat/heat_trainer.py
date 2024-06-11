import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import tensorlayerx as tlx
from gammagl.datasets import NGSIM_US_101
from gammagl.models import HEAT
from gammagl.loader import DataLoader
from tensorlayerx.model import TrainOneStep, WithLoss


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fun):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fun)

    def forward(self, data, label):
        logits = self._backbone(data.x, data.edge_index, data.edge_attr, data.edge_type)

        train_logits = tlx.gather(logits, data.tar_mask)
        train_y = tlx.gather(data.y, data.tar_mask)

        loss = self._loss_fn(train_logits, train_y, reduction='mean')
        loss = tlx.sqrt(loss)
        # loss_each_data = tlx.sqrt(tlx.losses.mean_squared_error(train_logits, train_y, reduction='mean'))

        return loss


def main(args):
    # load datasets
    train_set = NGSIM_US_101(root=args.data_path, name='train')
    val_set = NGSIM_US_101(root=args.data_path, name='val')
    test_set = NGSIM_US_101(root=args.data_path, name='test')

    trainDataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valDataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    testDataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    net = HEAT(args.hist_length, args.in_channels_node, args.out_channels, args.out_length,
               args.in_channels_edge_attr, args.in_channels_edge_type, args.edge_attr_emb_size,
               args.edge_type_emb_size, args.node_emb_size, args.heads, args.concat, args.dropout, args.leaky_rate)

    print('loading HEAT model')

    optimizer = tlx.optimizers.Adam(lr=args.lr)
    train_weights = net.trainable_weights
    scheduler = tlx.optimizers.lr.MultiStepDecay(learning_rate=args.lr, milestones=[1, 2, 4, 6, 10, 30, 40, 50, 60],
                                                 gamma=0.7, verbose=True)
    loss_fn = SemiSpvzLoss(net, tlx.losses.mean_squared_error)
    train_one_step = TrainOneStep(loss_fn, optimizer, train_weights)

    best_val_loss = 1000
    for epoch in range(args.n_epoch):
        train_loss_epo = 0.0
        net.set_train()
        for i, data in enumerate(trainDataloader):
            indices = tlx.arange(0, args.out_length)
            data.y = tlx.gather(data.y, indices, axis=1)
            data.y = tlx.reshape(data.y, (data.y.shape[0], -1))
            loss_each_data = train_one_step(data, data.y)
            train_loss_epo += loss_each_data

        train_loss_epoch = round(train_loss_epo * 0.3048 / (i + 1), 4)

        val_loss_epoch = 0
        net.set_eval()
        for j, data in enumerate(valDataloader):
            logits = net(data.x, data.edge_index, data.edge_attr, data.edge_type)
            indices = tlx.arange(0, args.out_length)
            data.y = tlx.gather(data.y, indices, axis=1)
            data.y = tlx.reshape(data.y, (data.y.shape[0], -1))
            val_logits = tlx.gather(logits, data.tar_mask)
            val_y = tlx.gather(data.y, data.tar_mask)
            val_loss_epoch += tlx.convert_to_numpy(
                tlx.sqrt(tlx.losses.mean_squared_error(val_logits, val_y, reduction='mean')))

        val_loss_epoch = round(val_loss_epoch * 0.3048 / (j + 1), 4)

        print("Epoch [{:0>3d}] ".format(epoch + 1) + "  train loss: {:.4f}".format(
            train_loss_epoch) + "  val loss: {:.4f}".format(val_loss_epoch))

        # save best model on evaluation set
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            net.save_weights(str(args.out_length) + '-' + str(best_val_loss) + '.npz', format='npz_dict')

        scheduler.step()

    # Euclidean distance
    net.set_eval()
    net.load_weights(str(args.out_length) + '-' + str(best_val_loss) + '.npz', format='npz_dict')
    total_distance = 0
    total_samples = 0

    for i, data in enumerate(testDataloader):
        logits = net(data.x, data.edge_index, data.edge_attr, data.edge_type)
        indices = tlx.arange(0, args.out_length)
        data.y = tlx.gather(data.y, indices, axis=1)
        data.y = tlx.reshape(data.y, (data.y.shape[0], -1))
        test_logits = tlx.gather(logits, data.tar_mask)
        test_y = tlx.gather(data.y, data.tar_mask)

        # Calculate Euclidean distance
        distance = tlx.sqrt(tlx.reduce_sum(tlx.square(test_logits - test_y)))
        total_distance += distance
        total_samples += len(test_logits)

        # print("Euclidean distance for batch {}: {:.4f}".format(i + 1, distance))

    # Calculate average Euclidean distance
    average_distance = total_distance / total_samples
    print("Average Euclidean distance: {:.4f}".format(average_distance))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=40, help="number of epoch")
    parser.add_argument("--in_channels_node", type=int, default=64, help="heat_in_channels_node")
    parser.add_argument("--in_channels_edge_attr", type=int, default=5, help="heat_in_channels_edge_attr")
    parser.add_argument("--in_channels_edge_type", type=int, default=6, help="heat_in_channels_edge_type")
    parser.add_argument("--edge_attr_emb_size", type=int, default=64, help="heat_edge_attr_emb_size")
    parser.add_argument("--edge_type_emb_size", type=int, default=64, help="heat_edge_type_emb_size")
    parser.add_argument("--node_emb_size", type=int, default=64, help="heat_node_emb_size")
    parser.add_argument("--out_channels", type=int, default=128, help="heat_out_channels")

    parser.add_argument("--heads", type=int, default=3, help="number of heads")
    parser.add_argument('--concat', type=bool, default=True, help='heat_concat')
    parser.add_argument("--hist_length", type=int, default=10, help="length of history trajectory")
    parser.add_argument("--out_length", type=int, default=30, help="length of future trajectory")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--leaky_rate", type=float, default=0.1, help="LeakyReLU rate")

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=20, help="batch")
    parser.add_argument("--data_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--result_path", type=str, default=r'', help="path to save result")
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    if args.device >= 0:
        tlx.set_device("GPU", args.device)
    else:
        tlx.set_device("CPU")

    main(args)

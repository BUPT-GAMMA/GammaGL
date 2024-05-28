import argparse
import os
import tensorlayerx as tlx
from gammagl.datasets import NGSIM_US_101
from gammagl.models import HEAT
from gammagl.loader import DataLoader


os.environ["OMP_NUM_THREADS"] = "4"
os.environ['TL_BACKEND'] = 'torch'


class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        if self.last_epoch in self.milestones:
            self.optimizer.lr = self.optimizer.lr * self.gamma

        self.last_epoch += 1


def main(parser_args):
    # load datasets
    dataset = NGSIM_US_101(save_to=parser_args.data_path, data_path=parser_args.data_path).download()

    train_set = NGSIM_US_101(data_path=f'{parser_args.data_path}/train')
    val_set = NGSIM_US_101(data_path=f'{parser_args.data_path}/val')

    trainDataloader = DataLoader(train_set, batch_size=parser_args.batch_size, shuffle=True)
    valDataloader = DataLoader(val_set, batch_size=parser_args.batch_size, shuffle=True)

    train_net = HEAT(parser_args)
    print('loading HEAT model')

    train_net.to(parser_args.device)

    optimizer = tlx.optimizers.Adam(lr=0.001)

    train_weights = train_net.trainable_weights

    scheduler = MultiStepLR(optimizer, milestones=[1, 2, 4, 6, 10, 30, 40, 50, 60], gamma=0.7)

    val_loss = []
    train_loss = []

    for epoch in range(parser_args.n_epoch):
        # print(epoch)
        train_loss_epo = 0.0
        train_net.set_train()
        for i, data in enumerate(trainDataloader):
            data.y = data.y[:, 0:parser_args.out_length, :]
            data.y = data.y.view(data.y.shape[0], -1)

            logits = train_net(data)
            train_logits = tlx.gather(logits, data.tar_mask)
            train_y = tlx.gather(data.y, data.tar_mask)
            # print(train_logits.shape, train_y.shape)
            loss_each_data = tlx.sqrt(tlx.losses.mean_squared_error(train_logits, train_y, reduction='mean'))

            grads = optimizer.gradient(loss_each_data, train_weights)
            optimizer.apply_gradients(grads_and_vars=grads)

            train_loss_epo += tlx.convert_to_numpy(loss_each_data)

        train_loss_epo = round(train_loss_epo * 0.3048 / (i + 1), 4)
        train_loss.append(train_loss_epo)
        # print('epoch:', train_loss[epoch])

        val_loss_epoch = 0
        train_net.set_eval()
        for j, data in enumerate(valDataloader):
            logits = train_net(data)

            data.y = data.y[:, 0:parser_args.out_length, :]
            data.y = data.y.view(data.y.shape[0], -1)
            val_logits = tlx.gather(logits, data.tar_mask)
            val_y = tlx.gather(data.y, data.tar_mask)

            val_loss_epoch += tlx.convert_to_numpy(
                tlx.sqrt(tlx.losses.mean_squared_error(val_logits, val_y, reduction='mean')))

        val_loss_epoch = round(val_loss_epoch * 0.3048 / (j + 1), 4)
        val_loss.append(val_loss_epoch)
        scheduler.step()

        # save model
        train_net.save_weights(str(val_loss_epoch) + '.npz', format='npz_dict')

        print("Epoch [{:0>3d}] ".format(epoch + 1) + "  train loss: {:.4f}".format(
            train_loss[epoch]) + "  val loss: {:.4f}".format(val_loss[epoch]))


if __name__ == '__main__':
    # device = tlx.set_device(device='cpu')
    device = tlx.set_device(device='GPU', id=0)
    # Network arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=20, help="number of epoch")
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

    parser.add_argument("--batch_size", type=int, default=20, help="batch")
    parser.add_argument("--data_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--result_path", type=str, default=r'', help="path to save result")
    parser.add_argument("--device", type=int, default=0)

    parser_args = parser.parse_args()

    main(parser_args)

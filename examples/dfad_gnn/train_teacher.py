import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import tensorlayerx as tlx
from gammagl.models import GINModel
from gammagl.loader import DataLoader
from gammagl.datasets import TUDataset
from tensorlayerx.model import TrainOneStep, WithLoss
import numpy 

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        train_logits = self.backbone_network(data['x'], data['edge_index'], data['batch'])
        loss = self._loss_fn(train_logits, data['y'])
        return loss

def train_teacher(args):
    dataset = TUDataset(args.dataset_path,args.dataset)
    
    dataset_unit = len(dataset) // 10
    train_set = dataset[4 * dataset_unit:]
    val_set = dataset[:2 * dataset_unit]
    test_set = dataset[2 * dataset_unit: 4 * dataset_unit]
    # train_set = dataset[2 * dataset_unit:]
    # val_set = dataset[:dataset_unit]
    # test_set = dataset[dataset_unit: 2 * dataset_unit]
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    teacher = GINModel(
        in_channels=max(dataset.num_features, 1),
        hidden_channels=args.hidden_dim,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        name="GIN"
    )
    
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = teacher.trainable_weights
    loss_func = SemiSpvzLoss(teacher, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        teacher.set_train()

        for data in train_loader:
            print(data.x.shape)
            train_loss = train_one_step(data, data.y)

        teacher.set_eval()

        total_correct = 0
        for data in val_loader:
            val_logits = teacher(data.x, data.edge_index, data.batch)
            pred = tlx.argmax(val_logits, axis=-1)
            total_correct += int((numpy.sum(tlx.convert_to_numpy(pred == data.y).astype(int))))
        val_acc = total_correct / len(val_set)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            teacher.save_weights("./teacher_" + args.dataset + ".npz", format='npz_dict')
    
    teacher.load_weights("./teacher_" + args.dataset + ".npz", format='npz_dict', skip=True)
    teacher.set_eval()
    total_correct = 0
    for data in test_loader:
        test_logits = teacher(data.x, data.edge_index, data.batch)
        pred = tlx.argmax(test_logits, axis=-1)
        total_correct += int((numpy.sum(tlx.convert_to_numpy(pred == data['y']).astype(int))))
    test_acc = total_correct / len(test_set)

    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=1000, help="number of epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    train_teacher(args)
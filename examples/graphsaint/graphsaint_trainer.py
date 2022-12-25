import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep,WithLoss
from gammagl.loader import GraphSAINTRandomWalkSampler
from gammagl.models import GraphSAINTModel
from gammagl.datasets import Flickr
import argparse

class SAINTLoss(WithLoss):
    def __init__(self,net,loss_fn):
        super(WithLoss,self).__init__(backbone=net,loss_fn=loss_fn)
    def forward(self,data,y):
        train_logits = self.backbone_network(data.x,data.edge_index,data.batch)
        loss = self._loss_fn(train_logits,data.y)
        return loss


def main(args):

    # load datasets
    path = args.dataset_path
    dataset = Flickr(path)
    dataset_num = len(dataset)
    train_dataset = dataset[:int(dataset_num*0.6)]
    val_dataset = dataset[int(dataset_num*0.6):int(dataset_num*0.8)]
    test_dataset = dataset[int(dataset_num*0.8):]
    
    # add args
    train_loader = GraphSAINTRandomWalkSampler()
    val_loader = GraphSAINTRandomWalkSampler()
    test_loader = GraphSAINTRandomWalkSampler()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tlx.set_device('GPU', 5)
    model = GraphSAINTModel(in_channels=max(dataset.num_node_features,1),
                            n_hiddens=args.hidden_dim,
                            out_channels=dataset.num_classes,
                            p_dropout=args.drop_rate)

    optimizer = tlx.optimizers.Adam(lr = args.lr,weight_decay = args.l2_coef)

    train_weights = model.trainable_weights

    loss_func = SAINTLoss(model,tlx.losses.softmax_cross_entropy_with_logits)

    train_one_step = TrainOneStep(loss_func,optimizer,train_weights)

    print("loading dataset...")
    
    # train
    for epoch in args.n_epoch:
        model.set_train()
        model.set_aggr('add' if args.use_normalization else 'mean')
        for data in train_loader:
            train_loss = train_one_step(data, data.y)

        model.set_eval()
        model.set_aggr('mean')
        total_correct = 0
        best_val_acc = 0
        for data in val_loader:
            val_logits = model(data.x,data.edge_index,None,data.num_nodes)
            pred = tlx.argmax(val_logits,axis=-1)
            total_correct += int((np.sum(tlx.convert_to_numpy(pred == data['y']).astype(int))))  
        val_acc = total_correct/len(val_dataset)
        
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
        #print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, 'f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')

    # test
    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    model.set_eval()
    total_correct = 0
    for data in test_loader:
        test_logits = model(data.x, data.edge_index, data.batch)
        # test_logits = net(data.x, data.edge_index, None, data.batch.shape[0], data.batch)
        pred = tlx.argmax(test_logits, axis=-1)
        total_correct += int((np.sum(tlx.convert_to_numpy(pred == data['y']).astype(int))))
    test_acc = total_correct / len(test_dataset)

    print("Test acc:  {:.4f}".format(test_acc))


def train(dataloader):
    model.set_train()
    model.set_aggr('add' if args.use_normalization else 'mean')
    for data in dataloader:
        train_loss = train_one_step(data,data.y)
    #     data = data.to(device)
    #     optimizer.zero_grad()
    #     if args.use_normalization:
    #         edge_weight=data.edge_norm * data.edge_weight
    #         output = model(data.x,data.edge_index,edge_weight)
    #         loss = 

def val(test_loader):
    model.set_eval()
    model.set_aggr('mean')
    total_correct = 0
    for data in test_loader:
        test_logits = model(data.x,data.edge_index,None,data.num_nodes)
        pred = tlx.argmax(test_logits,axis=-1)
        total_correct += int((np.sum(tlx.convert_to_numpy(pred == data['y']).astype(int))))  
    test_acc = total_correct/len(test_dataset)
    return test_acc




if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=32, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    # parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    # parser.add_argument('--dataset', type=str, default='IMDB-BINARY', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    # parser.add_argument('--dataset', type=str, default='REDDIT-BINARY', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    parser.add_argument("--dataset_path", type=str, default=r'../../../data', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--num_layers", type=int, default=5, help="num of gin layers")
    parser.add_argument("--batch_size", type=int, default=100, help="batch_size of the data_loader")
    parser.add_argument("--use_normalization",type=bool,default=True,help="whether use normalization or not")

    args = parser.parse_args()

    main(args)
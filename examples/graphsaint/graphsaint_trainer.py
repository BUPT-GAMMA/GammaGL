import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep,WithLoss
from gammagl.loader import DataLoader,GraphSAINT_Sampler
from gammagl.models import GraphSAINTModel
from gammagl.datasets import Flickr
from tqdm import tqdm
from gammagl.utils import degree,add_self_loops, mask_to_index
import argparse
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
class SAINTLoss(WithLoss):
    def __init__(self,net,loss_fn=None):
        super(SAINTLoss,self).__init__(backbone=net,loss_fn=loss_fn)
        self.net=net
    def forward(self,data,y):
        train_logits = self.backbone_network(data['x'],data['edge_index'])
        loss = self._loss_fn(train_logits,data['y'])
        return loss

def rename_index(edge_index):
    edge_index = edge_index.numpy()
    unique_values, indices = np.unique(edge_index, return_inverse=True)
    sorted_values = np.sort(unique_values)
    renamed_matrix = sorted_values.searchsorted(unique_values[indices]).reshape(edge_index.shape)
    return tlx.convert_to_tensor(renamed_matrix)

def main(args):

    # load datasets
    path = args.dataset_path
    # dataset = Flickr("../../../data")
    dataset = Flickr(path)
    # train_dataset = dataset[:int(dataset_num*0.6)]
    # val_dataset = dataset[int(dataset_num*0.6):int(dataset_num*0.8)]
    # test_dataset = dataset[int(dataset_num*0.8):]

    graph = dataset[0]
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)
    # add args
    # train_loader = GraphSAINTRandomWalkSampler()
    # val_loader = GraphSAINTRandomWalkSampler()
    # test_loader = GraphSAINTRandomWalkSampler()
    # loader = DataLoader(dataset, batch_size=args.batch_size)
   
    data = {
        "x": graph.x,
        "y":graph.y,
        "edge_index":graph.edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges
    }
    
    data['edge_weight'] = 1. / degree(data['edge_index'][1],data['num_nodes'])
    loader = GraphSAINT_Sampler(data,num_steps=5,sample_coverage=100,batch_size=2048,num_workers=0)
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
    for epoch in range(args.n_epoch):
        print("epoch ",epoch," / ",args.n_epoch)
        model.set_train()
        model.set_aggr('add' if args.use_normalization else 'mean')
        for batch in range(args.batch_size):
            for indptr,indices,value,subg_nodes,subg_edge_index in loader:
                traindata={
                "x":tlx.gather(tlx.convert_to_tensor(graph.x),tlx.convert_to_tensor(subg_nodes)),
                "y":tlx.gather(tlx.convert_to_tensor(graph.y),tlx.convert_to_tensor(subg_nodes)),
                #"edge_index":graph.edge_index 
                "edge_index":rename_index(tlx.gather(tlx.convert_to_tensor(graph.edge_index),tlx.convert_to_tensor(subg_edge_index),axis=1))
                }
                # rename edge_index
                # train_loss = train_one_step(data, data['y'])
                train_logits = model(traindata["x"],traindata["edge_index"])
                train_loss = train_one_step(traindata, tlx.convert_to_tensor([0]))
        print("train_loss : ",train_loss)
    model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
      

    # test
    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    model.set_eval()
    total_correct = 0
    total_num = 0
    for indptr,indices,value,subg_nodes,subg_edge_index in loader:
        testdata={
            "x":tlx.gather(tlx.convert_to_tensor(graph.x),tlx.convert_to_tensor(subg_nodes)),
            "y":tlx.gather(tlx.convert_to_tensor(graph.y),tlx.convert_to_tensor(subg_nodes)),
            #"edge_index":graph.edge_index 
            "edge_index":rename_index(tlx.gather(tlx.convert_to_tensor(graph.edge_index),tlx.convert_to_tensor(subg_edge_index),axis=1))
        }
        test_logits = model(testdata["x"],testdata["edge_index"])
        y_pred = tlx.argmax(test_logits, axis=-1)
        y_true = testdata['y']
        F1_score = f1_score(y_true,y_pred,average='micro')
        
    print("F1_score:  {:.4f}".format(F1_score))



if __name__ == '__main__':
    # parameters setting
    tlx.set_device('GPU', 5)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=20, help="number of epoch")
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
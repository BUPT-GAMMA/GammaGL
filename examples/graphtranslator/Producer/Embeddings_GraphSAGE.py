import time
from ogb.nodeproppred import PygNodePropPredDataset
import torch

from gammagl.data import Graph
from gammagl.loader import LinkNeighborLoader
from gammagl.layers.conv import SAGEConv
import tensorlayerx.nn as nn
import tensorlayerx as tlx
from tensorlayerx.model import WithLoss, TrainOneStep
import os
os.environ['TL_BACKEND'] = "torch"

bert_node_embeddings = torch.load("../data/bert_node_embeddings.pt")

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/') 

edge_index = dataset[0].edge_index
row, col = edge_index[0], edge_index[1]

src_node = tlx.concat((tlx.convert_to_tensor(row), tlx.convert_to_tensor(col)), axis=0)
dst_node = tlx.concat((tlx.convert_to_tensor(col), tlx.convert_to_tensor(row)), axis=0)
edge_index = tlx.stack((src_node, dst_node), axis=0)


split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

graph = Graph(x=bert_node_embeddings, edge_index=edge_index, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, y=dataset[0].y.squeeze())
train_loader = LinkNeighborLoader(
    graph,
    batch_size=65536,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[10, 10],
    edge_label_index=graph.edge_index,
    edge_label=None
)


class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_channels=in_dim,
                             out_channels=hid_dim)
        self.conv2 = SAGEConv(in_channels=hid_dim,
                             out_channels=out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Net(768, 1024, 768).to(device)

class LinkPredictionLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(LinkPredictionLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        h = self._backbone(data['x'], data['edge_index'])
        h_src = tlx.gather(h, data['edge_label_index'][0])
        h_dst = tlx.gather(h, data['edge_label_index'][1])
        pred = tlx.reduce_sum(h_src * h_dst, axis=-1)
        loss = self._loss_fn(output=pred, target=label)
        return loss

def train():

    optimizer = tlx.optimizers.Adam(0.01)
    loss_func = LinkPredictionLoss(model, tlx.losses.sigmoid_cross_entropy)
    train_one_step = TrainOneStep(loss_func, optimizer, model.trainable_weights)

    total_loss = 0
    total_num = 0
    for batch in train_loader:
        data = {'x': batch.x.to(device), 
            'edge_index': batch.edge_index.to(device),
            'edge_label_index': batch.edge_label_index.to(device)
        }
        model.set_train()
        loss = train_one_step(data, batch.edge_label.to(device))
        total_loss += float(loss) * batch.edge_label.shape[0]
        total_num += batch.edge_label.shape[0]

    return total_loss/ total_num


best_acc = 0
for epoch in range(10):
    start = time.time()
    loss = train()
    print("loss:", loss)

out = model(graph.x, graph.edge_index)
if os.environ['TL_BACKEND'] == "torch":
    torch.save(out, "../../data/graphsage_node_embeddings.pt")


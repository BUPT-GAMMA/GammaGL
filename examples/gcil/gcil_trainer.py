import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR


import numpy as np

from gammagl.models import GCILModel, LogReg
from gammagl.utils import  mask_to_index, add_self_loops
from aug import random_aug
from params import set_params

from gammagl.data import Graph
from gammagl.datasets import WikiCS, Flickr, Planetoid, Coauthor

from sklearn.metrics import f1_score
import scipy.sparse as sp
import pandas as pd
import warnings
import tensorlayerx as tlx
from tensorlayerx.optimizers import Adam
from tensorlayerx.model import TrainOneStep, WithLoss


warnings.filterwarnings('ignore')

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

def calculate_acc(logits, y, metrics):
    metrics.update(logits, y) 
    rst = metrics.result() 
    metrics.reset() 
    return rst


def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)
    K_ = sp.diags(1./dist)*K
    dist = dist.reshape(-1, 1)
    ll = 0
    for it in range(sin_iter):
        u = 1./K_.dot(dist / (K.T.dot(u)))
    v = dist / (K.T.dot(u))
    delta = np.diag(u.reshape(-1)).dot(K).dot(np.diag(v.reshape(-1)))
    return delta


def plug(theta, num_node, laplace, delta_add, delta_dele, epsilon, dist, sin_iter, c_flag=False):
    C = (1 - theta)*laplace.A
    if c_flag:
        C = laplace.A
    K_add = np.exp(2 * (C*delta_add).sum() * C / epsilon)
    K_dele = np.exp(-2 * (C*delta_dele).sum() * C / epsilon)

    delta_add = sinkhorn(K_add, dist, sin_iter)

    delta_dele = sinkhorn(K_dele, dist, sin_iter)
    return delta_add, delta_dele


def update(theta, epoch, total):
    theta = theta - theta*(epoch/total)
    return theta


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(np.abs(adj.A).sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = tlx.convert_to_numpy(tlx.reduce_sum(tlx.to_device(features, "cpu"), axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(tlx.to_device(features, "cpu"))
    # return tlx.convert_to_tensor(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)

def get_dataset(path,name,scope_flag):
    if name == 'cora':
        dataset = Planetoid(root="", name='cora')
    elif name == 'citeseer':
        dataset = Planetoid(root="", name='citeseer')
    elif name == 'pubmed':
        dataset = Planetoid(root="", name='pubmed')
    elif name == 'wiki':
        dataset = WikiCS("")
    elif name == 'flickr':
        dataset = Flickr("")
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    graph = dataset[0]

    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    scope = add_self_loops(edge_index)[0]
    adj = sp.coo_matrix((np.ones(tlx.get_tensor_shape(edge_index)[1]),
                          (tlx.to_device(edge_index[0, :], "cpu"), tlx.to_device(edge_index[1, :], "cpu"))),
                          shape=[num_nodes, num_nodes])


    feat = graph.x
    if name != 'wiki':
        feat = tlx.convert_to_tensor(preprocess_features(feat), dtype=tlx.float32)
    else:
        feat = tlx.convert_to_tensor(feat, dtype=tlx.float32)


    num_features = feat.shape[-1]
    label = graph.y
    num_class = label.max()+1

    idx_train = mask_to_index(graph.train_mask)
    idx_test = mask_to_index(graph.test_mask)
    idx_val = mask_to_index(graph.val_mask)

    laplace = sp.eye(adj.shape[0]) - normalize_adj(adj)
    
    return adj, feat, label, num_class, idx_train, idx_val, idx_test, laplace, scope


def test(embeds, labels, num_class, train_idx, val_idx, test_idx):

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    ''' Linear Evaluation '''
    # print(train_embs.shape)
    logreg = LogReg(train_embs.shape[1], num_class)
    opt = Adam(lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = tlx.losses.softmax_cross_entropy_with_logits

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(800):
        logreg.train()
        logits = logreg(train_embs)
        preds = tlx.argmax(logits, axis=1)
        train_acc = tlx.reduce_sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()


        logreg.eval()
        val_logits = logreg(val_embs)
        test_logits = logreg(test_embs)

        val_preds = tlx.argmax(val_logits, axis=1)
        test_preds = tlx.argmax(test_logits, axis=1)

        val_acc = tlx.reduce_sum(val_preds == val_labels).float() / val_labels.shape[0]
        test_acc = tlx.reduce_sum(test_preds == test_labels).float() / test_labels.shape[0]

        test_f1_macro = f1_score(
            test_labels.cpu(), test_preds.cpu(), average='macro')
        test_f1_micro = f1_score(
            test_labels.cpu(), test_preds.cpu(), average='micro')
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            if test_acc > eval_acc:
                test_f1_macro_ll = test_f1_macro
                test_f1_micro_ll = test_f1_micro

    print('Epoch:{}, train_acc:{:.4f}, Macro:{:4f}, Micro:{:4f}'.format(epoch, train_acc, test_f1_macro_ll,
                                                                        test_f1_micro_ll))

    return test_f1_macro_ll, test_f1_micro_ll


def train(params):
    path = "./dataset/" + args.dataset
    adj, feat, labels, num_class, train_idx, val_idx, test_idx, laplace, scope = get_dataset(path, args.dataset, args.scope_flag)

    adj = adj + sp.eye(adj.shape[0])

    edge_index = np.vstack((adj.nonzero()[0], adj.nonzero()[1]))
    x = tlx.convert_to_tensor(feat, dtype=tlx.float32)  
    graph = Graph(x=x, edge_index=tlx.convert_to_tensor(edge_index, dtype=tlx.int64))
    
    if args.dataset == 'pubmed':
        new_adjs = []
        for i in range(10):
            new_adjs.append(sp.load_npz(path + "/0.01_1_" + str(i) + ".npz"))
        adj_num = len(new_adjs)
        adj_inter = int(adj_num / args.num)
        sele_adjs = []
        for i in range(args.num + 1):
            try:
                if i == 0:
                    sele_adjs.append(new_adjs[i])
                else:
                    sele_adjs.append(new_adjs[i * adj_inter - 1])
            except IndexError:
                pass
        print("Number of select adjs:", len(sele_adjs))
        epoch_inter = args.epoch_inter
    elif args.dataset == 'wiki':
        sele_adjs = []
        for i in range(7):
            sele_adjs.append(sp.load_npz(path + "/0.1_1_" + str(i) + ".npz"))
        epoch_inter = args.epoch_inter
    elif args.dataset == 'cora':
        sele_adjs = []
        for i in range(7):
            sele_adjs.append(sp.load_npz(path + "/0.01_1_" + str(i) + ".npz"))
        epoch_inter = args.epoch_inter
    elif args.dataset == 'blog':
        sele_adjs = []
        for i in range(7):
            sele_adjs.append(sp.load_npz(path + "/0.01_1_" + str(i) + ".npz"))
        epoch_inter = args.epoch_inter
    elif args.dataset == 'flickr':
        sele_adjs = []
        for i in range(4):
            sele_adjs.append(sp.load_npz(path + "/0.01_1_" + str(i) + ".npz"))
        epoch_inter = args.epoch_inter
    else:
        scope_matrix = sp.coo_matrix(
            (np.ones(scope.shape[1]), (scope[0, :], scope[1, :])), shape=adj.shape).A
        dist = adj.A.sum(-1) / adj.A.sum()

    if args.gpu != -1:
        args.device = 'cuda:{}'.format(args.gpu) 
    else:
        args.device = 'cpu'

    in_dim = feat.shape[1]
    
    model = GCILModel(in_dim, args.hid_dim, args.out_dim, args.n_layers, args.use_mlp)
    model = model.to(args.device)

    optimizer = tlx.optimizers.Adam(
    lr=args.lr1,
    weight_decay=args.wd1
    )
    num_nodes = graph.num_nodes
    N = graph.num_nodes
    
    loss_fn = tlx.losses.softmax_cross_entropy_with_logits

    # loss_func = SemiSpvzLoss(net=model, loss_fn=loss_fn)
    # train_one_step = TrainOneStep(loss_func, optimizer, model.trainable_weights)

    #### SpCo ######
    theta = 1
    delta = np.ones(adj.shape) * args.delta_origin
    delta_add = delta
    delta_dele = delta
    num_node = adj.shape[0]
    range_node = np.arange(num_node)
    ori_graph = graph
    new_graph = ori_graph

    new_adj = adj.tocsc()
    ori_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()])[0]  
    ori_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node])[0] 
    new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()])[0]
    new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node])[0]
    j = 0

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "in_dim": in_dim,
        "num_nodes": num_nodes
    }
    
    # Training Loop
    for epoch in range(params['epoch']):
        model.train()
        # train_loss = train_one_step(data, graph.y)

        graph1_, attr1, feat1 = random_aug(new_graph, new_attr, new_diag_attr, feat, args.dfr, args.der, args.device)
        graph2_, attr2, feat2 = random_aug(ori_graph, ori_attr, ori_diag_attr, feat, args.dfr, args.der, args.device)

        graph1 = Graph(x=graph1_.x.to(args.device), edge_index=graph1_.edge_index.to(args.device))
        graph2 = Graph(x=graph2_.x.to(args.device), edge_index=graph2_.edge_index.to(args.device))

        attr1 = attr1.to(args.device)
        attr2 = attr2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        z1, z2, h1, h2 = model(graph1, feat1, attr1, graph2, feat2, attr2)
        # print(z1.shape, z2.shape, h1.shape, h2.shape)

        std_x = tlx.sqrt(h1.var(dim=0) + 0.0001)
        std_y = tlx.sqrt(h2.var(dim=0) + 0.0001)

        std_loss = tlx.ops.reduce_sum(tlx.sqrt((1 - std_x)**2)) / 2 + tlx.ops.reduce_sum(tlx.sqrt((1 - std_y)**2)) / 2

        c = tlx.matmul(z1.T, z2) / N
        c1 = tlx.matmul(z1.T, z1) / N
        c2 = tlx.matmul(z2.T, z2) / N

        loss_inv = -tlx.diag(c).sum() 
        iden = tlx.convert_to_tensor(np.eye(c.shape[0])).to(args.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()



        loss = params['alpha']*loss_inv + params['beta'] * \
            (loss_dec1 + loss_dec2) + params['gamma']*std_loss

        loss.backward()

        # Print the results
        print('Epoch={:03d}, loss={:.4f}, loss_inv={:.4f}, loss_dec={:.4f}'.format(
        epoch, 
        loss.detach().cpu().numpy().item(),  
        -tlx.diag(c).sum().detach().cpu().numpy().item(),  
        (iden - c1).pow(2).sum().detach().cpu().numpy().item() + (iden - c2).pow(2).sum().detach().cpu().numpy().item()  # 同样处理loss_dec
        ))


        if args.dataset == 'pubmed':
            if (epoch - 1) % epoch_inter == 0:
                try:
                    print("================================================")
                    delta = args.lam * sele_adjs[int(epoch / epoch_inter)]
                    new_adj = adj + delta

                    nonzero_indices = np.array(new_adj.nonzero())
                    edge_index = tlx.ops.convert_to_tensor(nonzero_indices, dtype=tlx.int64)
                    
                    x = tlx.convert_to_tensor(feat, dtype=tlx.float32)
                    new_graph = Graph(x=x, edge_index=edge_index)
                    
                    new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]  # 使用 tlx.float32
                    new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[0]  # 使用 tlx.float32

                except IndexError:
                    pass


        elif args.dataset in ['wiki', 'cora', 'blog', 'flickr']:
            flag = (epoch - 1) % epoch_inter
            if flag == 0:
                try:
                    print("================================================")
                    delta = args.lam * sele_adjs[(epoch - 1)//epoch_inter]
                    new_adj = adj + delta
                    
                    nonzero_indices = np.array(new_adj.nonzero())
                    edge_index = tlx.ops.convert_to_tensor(nonzero_indices, dtype=tlx.int64)

                    x = tlx.convert_to_tensor(feat, dtype=tlx.float32)
                    new_graph = Graph(x=x, edge_index=edge_index)

                    new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]  # 使用 tlx.float32
                    new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[0]  # 使用 tlx.float32
                except IndexError:
                    pass
        else:
            if epoch % args.turn == 0:
                print("================================================")
                if args.dataset in ["cora", "citeseer"] and epoch != 0:
                    delta_add, delta_dele = plug(theta, num_node, laplace, delta_add, delta_dele, args.epsilon, dist,
                                                args.sin_iter, True)
                else:
                    delta_add, delta_dele = plug(theta, num_node, laplace, delta_add, delta_dele, args.epsilon, dist,
                                                args.sin_iter)

                # 用 TensorLayerX 处理
                delta = (delta_add - delta_dele) * scope_matrix
                
                path_cora = path+'/0.01_1_'+str(j)+'.npz'
                sp.save_npz(path_cora, normalize_adj(delta))
                j += 1

                delta = args.lam * normalize_adj(delta)
                new_adj = adj + delta
                
                nonzero_indices = np.array(new_adj.nonzero())
                edge_index = tlx.ops.convert_to_tensor(nonzero_indices, dtype=tlx.int64)

                x = tlx.convert_to_tensor(feat, dtype=tlx.float32)
                new_graph = Graph(x=x, edge_index=edge_index)

                new_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]  # 使用 tlx.float32
                new_diag_attr = tlx.convert_to_tensor(new_adj[range_node, range_node], dtype=tlx.float32)[0]  # 使用 tlx.float32

                theta = update(1, epoch, args.epochs)


    # Final Testing
    # Remove direct use of graph.to(args.device) and replace it with moving the graph attributes manually
    graph.x = graph.x.to(args.device)  # Move features to device
    graph.edge_index = graph.edge_index.to(args.device)  # Move edge_index to device
    # Manually add self-loops without using remove_self_loop and add_self_loop methods
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index.cpu().numpy()  # Convert to numpy for edge_index
    self_loops = np.vstack([np.arange(num_nodes), np.arange(num_nodes)])
    edge_index = np.hstack([edge_index, self_loops])  # Add self loops to edge_index
    graph.edge_index = tlx.convert_to_tensor(edge_index, dtype=tlx.int64, device=args.device)  # Convert back to TensorLayerX tensor

    feat = feat.to(args.device)

    # Manually reconstruct adjacency matrix from edge_index
    new_adj = sp.coo_matrix((np.ones(graph.edge_index.shape[1]), 
                            (graph.edge_index[0].cpu().numpy(), graph.edge_index[1].cpu().numpy())),
                            shape=(num_nodes, num_nodes)).tocsc()

    # Reconstruct attributes from adjacency matrix
    attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32).to(args.device)

    # Get embeddings using the model
    embeds = model.get_embedding(graph, feat, attr)

    # Evaluation
    test_f1_macro_ll = 0
    test_f1_micro_ll = 0
    macros = []
    micros = []

    # Run the test function
    test_f1_macro_ll, test_f1_micro_ll = test(embeds, labels, num_class, train_idx, val_idx, test_idx)

    # Append results to macros and micros
    macros.append(test_f1_macro_ll)
    micros.append(test_f1_micro_ll)

    # Convert lists to tensors
    macros = tlx.convert_to_tensor(macros, dtype=tlx.float32)
    micros = tlx.convert_to_tensor(micros, dtype=tlx.float32)

    # Convert arguments to dictionary (if necessary)
    config = vars(args)


    config['test_f1_macro'] = test_f1_macro_ll
    config['test_f1_micro'] = test_f1_micro_ll
    print(config)

    return tlx.ops.reduce_mean(macros).cpu().numpy().item(), tlx.ops.reduce_mean(micros).cpu().numpy().item()



def main(args):
    print(args)
    macros, micros = [], []

    df = pd.DataFrame()
    params = {}
    params['alpha'], params['beta'], params['gamma'], params['epoch'] = 1, 0.015, 0, 100

    for i in range(10):
        ma, mi = train(params)
        macros.append(ma)
        micros.append(mi)
        print(ma, mi)
        print(params)

    print(params)

    micros = tlx.convert_to_tensor(micros)
    macros = tlx.convert_to_tensor(macros)

    print('AVG accuracy:{:.4f}, Std:{:.4f}, Macro:{:.4f}, Std:{:.4f}'.format(
    tlx.ops.reduce_mean(micros).cpu().numpy().item(),
    tlx.ops.reduce_std(micros).cpu().numpy().item(),
    tlx.ops.reduce_mean(macros).cpu().numpy().item(),
    tlx.ops.reduce_std(macros).cpu().numpy().item())
)




if __name__ == '__main__':
    args = set_params()

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)

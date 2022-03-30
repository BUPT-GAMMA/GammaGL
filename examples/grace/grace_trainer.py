from tqdm import tqdm
import tensorflow as tf
from gammagl.data import Graph
from gammagl.datasets import Planetoid
import tensorlayerx as tlx
import argparse
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from gammagl.models.grace import grace, calc
from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.utils.corrupt_graph import dfde_norm_g


class UnSemi_Loss(WithLoss):
    def __init__(self, net):
        super(UnSemi_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data['graph1'], data['graph2'])
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in
                            self._backbone.trainable_weights]) * args.l2  # only support for tensorflow backend
        return loss + l2_loss

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool8)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def label_classification(embeddings, y, train_mask, test_mask, split='random', ratio=0.1):
    x = embeddings.numpy();
    y = y.numpy()
    # y = y.reshape(-1, 1)
    y = y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(y)
    y = onehot_encoder.transform(y).toarray().astype(np.bool8)

    x = normalize(x, norm='l2')
    if split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 - ratio)
    elif split == 'public':
        X_train = x[train_mask]
        X_test = x[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    row, col, weight = calc(graph.edge_index, graph.num_nodes)
    orgin_graph = Graph(x=graph.x, edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64),
                        num_nodes=graph.num_nodes, y=graph.y)
    orgin_graph.edge_weight = weight
    x = graph.x

    # build model
    net = grace(in_feat=x.shape[1], hid_feat=args.hidden_dim,
                out_feat=args.hidden_dim,
                num_layers=args.num_layers,
                activation=tlx.nn.PRelu(args.hidden_dim),
                temp=args.temp)
    optimizer = tlx.optimizers.Adam(args.lr)
    train_weights = net.trainable_weights
    loss_func = UnSemi_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        graph1 = dfde_norm_g(graph.edge_index, x, args.drop_feature_rate_1, args.drop_edge_rate_1)
        graph2 = dfde_norm_g(graph.edge_index, x, args.drop_feature_rate_2, args.drop_edge_rate_2)
        data = {"graph1": graph1, "graph2": graph2}
        loss = train_one_step(data, label=None)
        print("loss :{:4f}".format(loss))
    print("=== Final ===")
    net.set_eval()
    embeds = net.get_embeding(graph=orgin_graph)
    '''Evaluation Embeddings  '''
    print(label_classification(embeds, tlx.argmax(graph.y, 1), graph.train_mask, graph.test_mask, args.split))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=256, help="dimention of hidden layers")
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.3)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2", type=float, default=1e-5, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='citeseer', help='dataset,cora/pubmed/citeseer')
    parser.add_argument('--split', type=str, default='public', help='random or public')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    main(args)

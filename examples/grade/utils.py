import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from gammagl.datasets import Planetoid,Amazon
import tensorlayerx as tlx

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {'mean': np.mean(values), 'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

@repeat(1)
def linear_clf(embeddings, y, train_mask, test_mask, degree, dataset):
    X = tlx.convert_to_numpy(embeddings)
    Y = tlx.convert_to_numpy(y)
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = Y[train_mask]
    y_test = Y[test_mask]
    degree = degree[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    acc = (np.argmax(y_test, axis=1) == np.argmax(y_pred, axis=1)).astype(float)
    degree_dict = {}
    for i in range(degree.shape[0]):
        if degree[i] not in degree_dict:
            degree_dict[degree[i]] = []
        degree_dict[degree[i]].append(acc[i])

    for d,l in degree_dict.items():
        degree_dict[d] = np.mean(l)
    bias = np.var(list(degree_dict.values()))
    mean = np.mean(list(degree_dict.values()))

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    return {'F1Mi': micro, 'F1Ma': macro, 'Mean':mean, 'Bias':bias}


def load(name, mode,path):
    assert name in ['cora', 'citeseer', 'photo', 'computers']
    assert mode in ['full', 'part']
    if name == 'cora':
        dataset = Planetoid(root=path,name='Cora')
        #idx_test = idx_test[1400:2400]
    elif name == 'citeseer':
        dataset = Planetoid(root=path,name='Citeseer')
        #idx_test = idx_test[:1000]

    elif name == 'photo':
        dataset = Amazon(root=path,name='Photo')
        #idx_test = idx_test[1000:2000]

    elif name == 'computers':
        dataset = Amazon(root=path,name='Computers')

    graph = dataset[0]
    num_nodes = graph.num_nodes
    edge_index=graph.edge_index
    feat = graph.x
    labels = graph.y
    degree=tlx.convert_to_numpy(graph.out_degree)
    num_class = dataset.num_classes

    # get nodes of which 0<degree<50
    idx_test = [i for i in range(num_nodes) if degree[i]>0 and degree[i]<50]
    #get 1000 nodes for testing
    if name=='cora':
        idx_test = idx_test[1400:2400]
    if name=='citeseer':
        idx_test = idx_test[:1000]
    if name=='photo':
        idx_test = idx_test[1000:2000]
    if name=='computers':
        idx_test = idx_test[2000:3000]

    # all nodes for training
    if mode == 'full':
        idx_train = [i for i in range(num_nodes) if i not in idx_test]
    # 50 nodes each class for training
    elif mode == 'part':
        idx_train = []
        for j in range(num_class):
            idx_train.extend([i for i,x in enumerate(labels) if x==j and i not in idx_test][:50])

    train_mask=np.zeros(shape=[num_nodes],dtype=bool)
    train_mask[idx_train]=True
    test_mask=np.zeros(shape=[num_nodes],dtype=bool)
    test_mask[idx_test]=True
    train_mask=tlx.convert_to_tensor(train_mask,dtype=tlx.bool)
    test_mask=tlx.convert_to_tensor(test_mask,dtype=tlx.bool)
    return edge_index, feat, labels, train_mask, test_mask, degree,num_nodes



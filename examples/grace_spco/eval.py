import numpy as np
import functools
import tensorlayerx as tlx

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
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


def label_classification(embeddings, y, idx_trains, idx_val, idx_test):
    X = tlx.convert_to_numpy(tlx.to_device(embeddings, "cpu"))
    Y = tlx.convert_to_numpy(tlx.to_device(y, "cpu"))
    idx_trains = tlx.to_device(idx_trains, "cpu")
    idx_test = tlx.to_device(idx_test, "cpu")
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')
    
    X_train = X[idx_trains]
    X_test = X[idx_test]
    y_train = Y[idx_trains]
    y_test = Y[idx_test]
    
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                        param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                        verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    print('F1Mi ', micro,'F1Ma ', macro)
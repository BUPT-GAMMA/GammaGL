import math
import numpy as np
import tensorlayerx as tlx
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

"=========== loss cal fnc =================="


def linearsvc(embeds, labels):
    x = embeds.numpy()
    y = labels.numpy()
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies)

def get_positive_expectation(p_samples, average=True):
    """
        Computes the positive part of a JS Divergence.
        
        Parameters
        ----------
        p_samples:
            Positive samples.
        average:
            Average the result over samples.

        Returns
        -------
        tensor
            Ep: mean of positive expectation.
    """
    log_2 = math.log(2.)
    Ep = log_2 - tlx.softplus(-p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Parameters
    ----------
    q_samples:
        Negative samples.
    average:
        Average the result over samples.

    Returns
    -------
    tensor
        Ep: Mean of negative expectation.
    """
    log_2 = math.log(2.)
    Eq = tlx.softplus(-q_samples) + q_samples - log_2
    if average:
        return Eq.mean()
    else:
        return Eq



def local_global_loss_(l_enc, g_enc, batch):
    """
    Computes the loss of the model.

    Parameters
    ----------
    l_enc:
        Local feature map
    g_enc:
        global features
    batch:
        the batch size of dataset

    Returns
    -------
    tensor
        loss: the loss of model

    """

    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = np.zeros((num_nodes, num_graphs), dtype=np.float32)
    neg_mask = np.ones((num_nodes, num_graphs), dtype=np.float32)

    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = tlx.matmul(l_enc, tlx.transpose(g_enc))

    if tlx.BACKEND == 'torch':
        pos_mask = tlx.convert_to_tensor(pos_mask).to(g_enc.device)
        neg_mask = tlx.convert_to_tensor(neg_mask).to(g_enc.device)
    else:
        pos_mask = tlx.convert_to_tensor(pos_mask)
        neg_mask = tlx.convert_to_tensor(neg_mask)

    E_pos = tlx.reduce_sum(get_positive_expectation(res * tlx.convert_to_tensor(pos_mask), average=False))
    E_pos = E_pos / num_nodes
    E_neg = tlx.reduce_sum(get_negative_expectation(res * tlx.convert_to_tensor(neg_mask), average=False))
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    loss = E_neg - E_pos

    return loss



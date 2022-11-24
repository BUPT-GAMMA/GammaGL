from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
'''
Code adapted from https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/evaluate_embedding.py
Linear evaluation on learned node embeddings
'''

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data)
        loss = self._loss_fn(logits, label)
        return loss


class LogReg(tlx.nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = tlx.nn.Linear(out_features=nb_classes, in_features=ft_in,
                                W_init=tlx.initializers.xavier_uniform(nb_classes))

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y, search):
    nb_classes = np.unique(y).shape[0]
    hid_units = x.shape[1]
    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = tlx.convert_to_tensor(train_embs), tlx.convert_to_tensor(train_lbls)
        test_embs, test_lbls = tlx.convert_to_tensor(test_embs), tlx.convert_to_tensor(test_lbls)

        log = LogReg(hid_units, nb_classes)
        optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=0.0)
        train_weights = log.trainable_weights
        loss_func = SemiSpvzLoss(log, tlx.losses.softmax_cross_entropy_with_logits)
        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
        best = 1e9
        for it in range(100):
            # log.set_train()
            loss = train_one_step(train_embs, train_lbls)
            if loss < best:
                best = loss
                log.save_weights(r'./' + "log.npz", format='npz_dict')
        log.load_weights(r'./' + 'log.npz', format='npz_dict')
        logits = log(test_embs)
        preds = tlx.argmax(logits, axis=-1)
        result = np.array((preds == test_lbls), dtype=np.int)
        acc = tlx.reduce_sum(result / test_lbls.shape[0])
        accs.append(acc.numpy())
    return np.mean(accs)


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)


def evaluate_embedding(embeddings, labels, method, search=True):
    """
    Computes the accuracy of the model.
    Args:
        embeddings: x
        labels: label
        method: four type of method to compute
        (logistic regression, svc, linearsvc, randomforest)

    Returns:
        accuracy: the accuracy of model
    """

    labels = preprocessing.LabelEncoder().fit_transform(labels.cpu())
    x = np.array(embeddings.cpu())
    y = np.array(labels)
    if method == 'log':
        classify = logistic_classify
    elif method == 'svc':
        classify = svc_classify
    elif method == 'linsvc':
        classify = linearsvc_classify
    elif method == 'rf':
        classify = randomforest_classify

    accuracy = classify(x, y, search)
    print(method, accuracy)
    return accuracy

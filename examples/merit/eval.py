from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
import tensorlayerx as tlx
import numpy as np
def evaluation(adj,adj_weight, diff,diff_weight, feat, gnn, idx_train, idx_test, sparse,labels,act):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    #c = 2.0 ** np.arange(-10, 10)

    #clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
    #                   param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
    #                   verbose=0)
    model = gnn
    #model.load_state_dict(gnn.state_dict())
    labels=labels.numpy()
    labels = labels.reshape(-1, 1)
    tlx.nn.PRelu
    embeds1 = model(feat, adj,adj_weight,feat.shape[0])
    embeds1 = act(embeds1).numpy()
    embeds2 = model(feat, diff, diff_weight,feat.shape[0])
    embeds2 = act(embeds2).numpy()
    embeds1=np.squeeze(embeds1)
    embeds2=np.squeeze(embeds2)
    train_embs = embeds1[idx_train] + embeds2[idx_train]
    test_embs = embeds1[idx_test] + embeds2[idx_test]
    train_labels = labels[idx_train]
    test_labels = labels[idx_test]
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    micro = f1_score(test_labels, pred_test_labels, average="micro")
    return accuracy_score(test_labels, pred_test_labels),micro

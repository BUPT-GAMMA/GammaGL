from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
import tensorlayerx as tlx
import numpy as np
def evaluation(adj,adj_weight, diff,diff_weight, feat, gnn, idx_train, idx_test,labels,act):
    clf = LogisticRegression(random_state=0, max_iter=2000)        
    model = gnn

    labels=labels.numpy()
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
    return accuracy_score(test_labels, pred_test_labels)

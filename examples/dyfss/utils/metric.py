from sklearn import metrics
from munkres import Munkres
import numpy as np

from sklearn.manifold import TSNE
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def cluster_accuracy(pred, labels, num):
    l1 = l2 = range(num)
    cost = np.zeros((num, num), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(labels) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(labels, new_predict)
    f1_macro = metrics.f1_score(labels, new_predict, average='macro')
    precision_macro = metrics.precision_score(labels, new_predict, average='macro')
    recall_macro = metrics.recall_score(labels, new_predict, average='macro')
    f1_micro = metrics.f1_score(labels, new_predict, average='micro')
    precision_micro = metrics.precision_score(labels, new_predict, average='micro')
    recall_micro = metrics.recall_score(labels, new_predict, average='micro')
    nmi = metrics.normalized_mutual_info_score(labels, pred, average_method='geometric')
    return acc, nmi, f1_macro


def negative_labels_acc(cluster_num, pseudo_nl_mask, gt_nl_targets):

    nl_gt_list = np.array(gt_nl_targets)
    pseudo_nl_mask = np.array(pseudo_nl_mask)
    one_hot_targets = np.eye(cluster_num)[nl_gt_list]
    one_hot_targets = one_hot_targets - 1
    one_hot_targets = np.abs(one_hot_targets)

    flat_pseudo_nl_mask = pseudo_nl_mask.reshape(1, -1)[0]
    flat_one_hot_targets = one_hot_targets.reshape(1, -1)[0]
    flat_one_hot_targets = flat_one_hot_targets[np.where(flat_pseudo_nl_mask == 1)]
    flat_pseudo_nl_mask = flat_pseudo_nl_mask[np.where(flat_pseudo_nl_mask == 1)]

    nl_accuracy = (flat_pseudo_nl_mask == flat_one_hot_targets) * 1
    nl_accuracy_final = (sum(nl_accuracy) / len(nl_accuracy)) * 100
    return nl_accuracy_final


def compute_average_nmi(new_labels_list):
    '''
        compute the average nmi of the each ssl cluster results
        the element of new_labels_list is array
    '''
    nmi_average_list = []
    for i in range(len(new_labels_list)):
        nmi_list = []
        for j in range(len(new_labels_list)):
            if i != j:
                nmi = metrics.normalized_mutual_info_score(new_labels_list[i], new_labels_list[j], average_method='geometric')
                nmi_list.append(nmi)
        nmi_ave = np.mean(nmi_list)
        nmi_average_list.append(nmi_ave)
    return nmi_average_list


def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.)
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=False,
                     fmt='.2f',
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()


# plot the scatter
def plot(X, fig, col, size, true_labels):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], lw=0, s=size, c=col[true_labels[i]])


def plotClusters(hidden_emb, true_labels, figure_name):
    print('Start plotting using TSNE...')
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    # Plot figure
    fig = plt.figure()
    plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink',
                            'orange', "olive", "cyan"], 10, true_labels)
    plt.axis("off")
    fig.savefig(figure_name, dpi=120)
    print("Finished plotting")

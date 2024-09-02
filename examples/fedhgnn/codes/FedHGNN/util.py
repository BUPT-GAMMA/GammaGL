#coding:utf-8
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.sparse
import logging

def print_log(file):
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别，可以根据需要调整
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.StreamHandler(),  # 输出到终端
            logging.FileHandler(file, mode='w'),  # 输出到文件
        ]
    )
    # 输出日志信息
    #logging.debug('信息将同时输出到终端和文件。')
    logging.info('信息会同时显示在终端和文件中。')


def pca_reduce(data, dim):
    pca = PCA(n_components=dim)
    pca = pca.fit(data)
    x = pca.transform(data)
    return x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


'''cluster'''
def cluster(feature_list, n_clusters):
    s = KMeans(n_clusters=n_clusters).fit(feature_list)
    #print(len(s.cluster_centers_))
    #每个样本所属的簇
    #print(len(s.labels_))
    label_count = {}
    for i in s.labels_:
        if(i not in label_count.keys()):
            label_count[i] = 1
        else:
            label_count[i]+=1

    #print(label_count)
    #print(s.labels_)

    label_author = {}
    author_label = {}
    labels = []
    for i, k in enumerate(s.labels_):
        author = i
        label = k
        labels.append(label)

        author_label[author] = label

        if(label not in label_author.keys()):
            label_author[label] = [author]
        else:
            label_author[label].append(author)

    # with open("./data_event/author_label", "w") as f:
    #     for l in author_label:
    #         f.write(l[0] + '\t' + l[1] + '\n')
    return label_count, labels, label_author, author_label
#cluster()

'''get shared knowledge rep(每个shared HIN的表示为它所包含的item的表示的平均）'''
def get_shared_knowledge_rep(item_feature_list, label_author):
    shared_knowledge_rep = {}
    for label, author_list in label_author.items():
        features = item_feature_list[author_list]
        rep = np.mean(features, 0)
        # sum = np.array([0.0]*len(item_feature_list[0]))
        # l = len(author_list)
        # for author in author_list:
        #     sum+= item_feature_list[author]
        # rep = sum/l
        shared_knowledge_rep[label] = rep
    return shared_knowledge_rep







def tsne(feature_list):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(feature_list)
    #print(tsne.embedding_)

    feature_list = tsne.embedding_
    print(np.shape(feature_list))#14795,2

    x = feature_list[:,0]
    y = feature_list[:,1]

    return x, y

#l = sio.loadmat("./data_event/author_tsne.mat")
# x= l['x']
# y =l['y']


def plot_embedding_2d(x, y, labels):
    """Plot an embedding X with the class label y colored by the domain d."""
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    plt.scatter(x, y, c=labels)

    # plt.xlim((-1.5, 1.5))
    # plt.xticks([])  # ignore xticks
    # plt.ylim((-1.5, 1.5))
    # plt.yticks([])  # ignore yticks
    plt.show()

#plot_embedding_2d(x,y)


def gen_shared_knowledge(adj, group_num):
    p_vs_f = adj[0]#(4025,73)
    p_vs_a = adj[1]#(4025,17431)
    p_vs_t = adj[2]#(4025,1903)
    p_vs_c = adj[3]#CSC (4025, 14)
    a_vs_t = p_vs_a.T * p_vs_t
    a_vs_f = p_vs_a.T * p_vs_f
    a_vs_c = p_vs_a.T * p_vs_c
    a_vs_p = p_vs_a.T
    a_vs_t_dense = a_vs_t.todense()
    a_vs_f_dense = a_vs_f.todense()
    a_vs_c_dense = a_vs_c.todense()
    a_vs_p_dense = a_vs_p.todense()
    #print(np.sum(a_vs_c_dense.sum(-1)==0))#大部分(10264)=0
    #print(a_vs_t_dense[1])
    a_feature = np.concatenate([a_vs_c_dense], -1)
    label_count, labels, label_author, author_label = cluster(a_feature, group_num) #20
    # x,y = tsne(a_feature)
    # plot_embedding_2d(x, y, labels)
    shared_knowledge_rep = get_shared_knowledge_rep(a_feature, label_author)
    return label_count, labels, label_author, author_label, shared_knowledge_rep

# if __name__ == 'main':
#     # feature_list = []
#     # for index in author_id_list:#
#     #     fea = features[index]
#     #     #print(len(fea))
#     #     feature_list.append(fea)
#     # feature_list = np.array(feature_list)

import numpy as np
import numpy as np
import warnings
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import tensorlayerx as tlx
import tensorlayerx.nn as nn
import tensorlayerx.optimizers as optim
import os

op1 = []
op2 = []
op3 = []

entity_count, relation_count = 0, 0
with open("./PubMed/node.dat", "r") as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split("\t")
        op1.append(temp1)
        op2.append(temp2)  # name
        op3.append(temp3)


def load(emb_file_path):

    emb_dict = {}
    with open(emb_file_path, "r") as emb_file:
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split("\t")
                emb_dict[index] = np.array(emb.split()).astype(np.float32)

    return train_para, emb_dict


tlx.set_device("GPU")
print("TL_BACKEND:", os.environ.get("TL_BACKEND"))

emb_file_path = "./PubMed/emb.dat"
train_para, emb_dict = load(emb_file_path)
link_test_file = "./PubMed/sample.dat"
pos_edges = []
neg_edges = []
with open(link_test_file, "r") as original_meta_file:
    for line in original_meta_file:
        temp = []
        temp1, temp2, temp3 = line[:-1].split("\t")
        # print(temp3=='1')
        temp.append(int(temp1))
        temp.append(int(temp2))
        if temp3 == "1":
            pos_edges.append(temp)
        if temp3 == "0":
            neg_edges.append(temp)
pos_edges_tensor = tlx.convert_to_tensor(pos_edges, dtype=tlx.int64)
neg_edges_tensor = tlx.convert_to_tensor(neg_edges, dtype=tlx.int64)
node_emb = []
d2l = dict()
line = 0
for key, values in emb_dict.items():
    node_emb.append(values)
    d2l[key] = line
    line = line + 1
node_emb = np.array(node_emb)
node_emb = tlx.convert_to_tensor(node_emb, dtype=tlx.float32)


class LMNN(nn.Module):
    def __init__(self, node_emb, pos_edges_tensor, neg_edges_tensor):
        super().__init__()
        vocab_size, embedding_dim = node_emb.shape
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.embeddings = nn.Parameter(node_emb)
        self.margin = 50
        self.pos_edges_tensor = pos_edges_tensor
        self.neg_edges_tensor = neg_edges_tensor

    def decode(self, h_all, idx):
        h = h_all

        emb_node1 = h[idx[:, 0]]
        emb_node2 = h[idx[:, 1]]
        # emb_node1 = gain_emb_tensor(h,idx[:, 0].tolist())
        # emb_node2 = gain_emb_tensor(h,idx[:, 1].tolist())
        # Rai=torch.sum(emb_node1*emb_node2, dim=-1, keepdim=True)
        sqdist = emb_node1 * emb_node2
        sqdist = tlx.reduce_sum(sqdist, axis=1)
        return sqdist

    def forward(self):
        pos_scores = self.decode(self.embeddings.embeddings, self.pos_edges_tensor)
        neg_scores = self.decode(self.embeddings.embeddings, self.neg_edges_tensor)
        loss = neg_scores - pos_scores + self.margin
        # print(pos_scores)
        # print(neg_scores)
        loss[loss < 0] = 0
        loss = tlx.reduce_sum(loss)
        # loss = -torch.mean(torch.log(torch.clamp(torch.sigmoid(Rai - Raj), min=1e-10, max=1.0)))
        return loss


model = LMNN(node_emb, pos_edges_tensor, neg_edges_tensor)
# print(model.get_embedding)
optimizer = optim.Adam(lr=0.001, grad_clip=tlx.ops.ClipGradByNorm(1.0))

for epoch in range(1000):
    model.train()
    train_loss = model()
    train_weights = model.trainable_weights
    grad = optimizer.gradient(loss=train_loss, weights=train_weights, return_grad=True)
    optimizer.apply_gradients(zip(grad, train_weights))

emb_trained = model.embeddings.embeddings
emb_trained_list = tlx.convert_to_numpy(emb_trained)
with open("./PubMed/sample_emb.dat", "w") as file:
    file.write("pubmed41\n")
    for i in range(len(op2)):
        file.write(f"{i}\t")
        file.write(" ".join(emb_trained_list[i].astype(str)))
        file.write("\n")


emb_file_path = "./PubMed/sample_emb.dat"
train_para, emb_dict = load(emb_file_path)


class MLP_Decoder(nn.Module):
    def __init__(self, hdim, nclass):
        super(MLP_Decoder, self).__init__()
        self.hidden_layer = nn.Linear(in_features=hdim, out_features=128)
        self.relu = nn.activation.ReLU()
        self.final_layer = nn.Linear(in_features=128, out_features=nclass)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.hidden_layer(h)
        h = self.relu(h)
        output = self.sigmoid(self.final_layer(h))
        return output


class Disease_MLP(nn.Module):
    def __init__(self, disease_dim, n_class):
        super(Disease_MLP, self).__init__()
        self.decoder = MLP_Decoder(disease_dim, n_class)

    def forward(self, dist):
        pred = self.decoder(dist)
        return pred


def mrrc(pred, label):
    n = len(pred)
    # print(pred)
    # print(label)
    sorted_idx = np.argsort(pred)[::-1]
    ranks = np.zeros(n)
    for i in range(n):
        ranks[sorted_idx[i]] = i + 1
    pos = np.where(label == 1)[0]
    if len(pos) == 0:
        return 0
    pos_rank = ranks[pos]
    return 1.0 / pos_rank


def lp_evaluate(test_file_path, emb_dict):

    posi, nega = defaultdict(set), defaultdict(set)
    with open(test_file_path, "r") as test_file:
        for line in test_file:
            left, right, label = line[:-1].split("\t")
            if label == "1":
                posi[left].add(right)
            elif label == "0":
                nega[left].add(right)

    edge_embs, edge_labels = defaultdict(list), defaultdict(list)
    for left, rights in posi.items():
        for right in rights:
            edge_embs[left].append(emb_dict[left] * emb_dict[right])
            edge_labels[left].append(1)
    for left, rights in nega.items():
        for right in rights:
            edge_embs[left].append(emb_dict[left] * emb_dict[right])
            edge_labels[left].append(0)

    for node in edge_embs:
        edge_embs[node] = np.array(edge_embs[node])
        edge_labels[node] = np.array(edge_labels[node])

    auc, mrr = cross_validation(edge_embs, edge_labels)

    return auc, mrr


def cross_validation(edge_embs, edge_labels):

    auc, mrr = [], []
    seed_nodes, num_nodes = np.array(list(edge_embs.keys())), len(edge_embs)
    # clf = Disease_MLP(768,1)tlx.convert_to_tensor
    seed = 1
    # optimizer = torch.optim.Adam(clf.parameters(), lr = 0.001)
    skf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros((num_nodes, 1)), np.zeros(num_nodes))
    ):

        print(f"Start Evaluation Fold {fold}!")
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = (
            [],
            [],
            [],
            [],
        )
        for each in train_idx:
            train_edge_embs.append(edge_embs[seed_nodes[each]])
            train_edge_labels.append(edge_labels[seed_nodes[each]])
        for each in test_idx:
            test_edge_embs.append(edge_embs[seed_nodes[each]])
            test_edge_labels.append(edge_labels[seed_nodes[each]])
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = (
            np.concatenate(train_edge_embs),
            np.concatenate(test_edge_embs),
            np.concatenate(train_edge_labels),
            np.concatenate(test_edge_labels),
        )
        best_auc = 0
        best_mrr = 0
        clf = Disease_MLP(768, 1)
        optimizer = optim.Adam(lr=0.001)
        for i in range(1000):
            clf.train()
            criterion = tlx.losses.binary_cross_entropy
            pred = tlx.ops.squeeze(clf(tlx.convert_to_tensor(train_edge_embs)), axis=1)
            # criterion = nn.BCEWithLogitsLoss()
            # loss = criterion(pred, truth.argmax(dim=1))
            train_edge_labels
            loss = criterion(
                pred, tlx.convert_to_tensor(train_edge_labels, tlx.float32)
            )
            train_weights_clf = clf.trainable_weights
            grad = optimizer.gradient(
                loss=loss, weights=train_weights_clf, return_grad=True
            )
            optimizer.apply_gradients(zip(grad, train_weights_clf))
            clf.set_eval()
            # clf.fit(train_edge_embs, train_edge_labels)
            preds = tlx.to_device(
                clf(tlx.convert_to_tensor(test_edge_embs)), device="CPU"
            )
            auc1 = roc_auc_score(test_edge_labels, preds)

            confidence = tlx.to_device(tlx.squeeze(preds, axis=1), device="CPU")
            # mrr1=np.mean(mrrc(np.array(confidence),test_edge_labels))

            curr_mrr, conf_num = [], 0
            for each in test_idx:
                test_edge_conf = np.argsort(
                    -confidence[
                        conf_num : conf_num + len(edge_labels[seed_nodes[each]])
                    ]
                )
                rank = np.empty_like(test_edge_conf)
                rank[test_edge_conf] = np.arange(len(test_edge_conf))
                curr_mrr.append(
                    1
                    / (
                        1
                        + np.min(
                            rank[
                                np.argwhere(
                                    edge_labels[seed_nodes[each]] == 1
                                ).flatten()
                            ]
                        )
                    )
                )
                conf_num += len(rank)

            mrr1 = np.mean(curr_mrr)
            if auc1 > best_auc:
                best_auc = auc1
                best_mrr = mrr1
            if i + 1 % 500 == 0:
                print(
                    "epoch:",
                    i,
                    "loss:",
                    loss.item(),
                    "auc:",
                    best_auc,
                    "mrr:",
                    best_mrr,
                )
        auc.append(best_auc)
        # print(tlx.convert_to_tensor(test_edge_labels)tlx.convert_to_tensor)
        mrr.append(best_mrr)
    print(auc)
    print(mrr)
    return np.mean(auc), np.mean(mrr)


link_test_file = "./PubMed/link.dat.test"
link_test_path = f"{link_test_file}"
scores = lp_evaluate(link_test_path, emb_dict)
print(scores)

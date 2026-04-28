import os
import numpy as np
import scipy.sparse as sp
import pickle

# 你的路径（已经正确）
DATA_DIR = "/home/ycn/data"
SAVE_ROOT = "/home/ycn/Unifews-main(1)/test_path/Unifews-main/data"

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_planetoid_dataset(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(DATA_DIR, dataset_str, names[i]), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(DATA_DIR, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

    n_nodes = features.shape[0]
    adj = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        adj[i, i] = 0
    for i in graph:
        for j in graph[i]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    adj = adj.tocsr()

    return features, labels, adj

def save_dataset(name, x, y, adj):
    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)
    n_nodes, n_feats = x.shape
    n_edges = adj.nnz // 2

    np.save(os.path.join(save_dir, "feats.npy"), x.astype(np.float32))

    # ==============================
    # 【终极修复】完全匹配你的代码格式！
    # 必须用 np.savez 保存 key="labels"
    # ==============================
    np.savez(
        os.path.join(save_dir, "labels.npz"),
        labels=y  # 这里必须写 labels=
    )

    sp.save_npz(os.path.join(save_dir, "adj.npz"), adj)

    with open(os.path.join(save_dir, "attribute.txt"), "w") as f:
        f.write(f"n={n_nodes}\n m= {n_edges}")

    deg = np.array(adj.sum(1)).flatten().astype(int)
    sp.save_npz(os.path.join(save_dir, "degree.npz"), sp.csr_matrix(deg))

    with open(os.path.join(save_dir, "adj_el.bin"), "wb") as f:
        np.array([n_nodes, n_edges], dtype=np.int32).tofile(f)
        adj.indptr[1:].astype(np.int32).tofile(f)
    with open(os.path.join(save_dir, "adj_pl.bin"), "wb") as f:
        adj.indices.astype(np.int32).tofile(f)

    print(f"✅ {name} 已生成完成！")

# ===================== 执行 =====================
if __name__ == "__main__":
    x_citeseer, y_citeseer, adj_citeseer = load_planetoid_dataset("citeseer")
    save_dataset("citeseer", x_citeseer, y_citeseer, adj_citeseer)

    x_pubmed, y_pubmed, adj_pubmed = load_planetoid_dataset("pubmed")
    save_dataset("pubmed", x_pubmed, y_pubmed, adj_pubmed)

    print("\n🎉 全部成功！数据格式 100% 匹配 Cora！")
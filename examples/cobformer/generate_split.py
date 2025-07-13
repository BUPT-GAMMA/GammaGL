import os
import os.path as osp
import numpy as np

def generate_splits(dataset, name, data_dir):
    """
    dataset: 已加载的 NCDataset，必须含 dataset.graph['num_nodes']
    name: 'film' 或 'deezer'
    data_dir: 存放 .npz 文件的目录
    """
    N = dataset.graph['num_nodes']
    os.makedirs(data_dir, exist_ok=True)

    for i in range(10):
        # 每一折用不同的 seed 保证可复现
        rng = np.random.default_rng(seed=i)
        perm = rng.permutation(N)
        
        n_train = int(N * 0.50)
        n_val   = int(N * 0.25)
        # 剩下的就是 test
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]
        
        # 构造布尔 mask
        train_mask = np.zeros(N, dtype=bool)
        val_mask   = np.zeros(N, dtype=bool)
        test_mask  = np.zeros(N, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True
        
        # 保存 npz
        out_path = osp.join(data_dir, f"{name}_split_50_25_{i}.npz")
        np.savez(out_path,
                 train_mask=train_mask,
                 val_mask=val_mask,
                 test_mask=test_mask)
        print(f"Saved split {i} ➜ {out_path}")

if __name__=="__main__":
    from Data.get_data import load_geom_gcn_dataset, load_deezer_dataset

    path = osp.join(osp.expanduser('~'), 'datasets', 'film')

    dataset = load_geom_gcn_dataset(path, "film")

    generate_splits(dataset, name="film", data_dir=path)

    path = osp.join(osp.expanduser('~'), 'datasets', 'deezer')

    dataset = load_deezer_dataset(path)

    generate_splits(dataset, name="deezer", data_dir=path)


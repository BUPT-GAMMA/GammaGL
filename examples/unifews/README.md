# Unifews
This is the original code for *Unifews: Unified Entry-Wise Sparsification for Efficient Graph Neural Network*, ICML 2025.

[Conference (Poster/Video/Slides)](https://icml.cc/virtual/2025/poster/45740) | [OpenReview](https://openreview.net/forum?id=INg866tEaT) | [arXiv](https://arxiv.org/abs/2403.13268) | [GitHub](https://github.com/gdmnl/Unifews)

### Citation

If you find this work useful, please cite our paper:
>  Ningyi Liao, Zihao Yu, Ruixiao Zeng, and Siqiang Luo.  
>  Unifews: You Need Fewer Operations for Efficient Graph Neural Networks.  
>  In Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 2025.
```
@inproceedings{liao2025unifews,
  title={{Unifews}: You Need Fewer Operations for Efficient Graph Neural Networks},
  author={Liao, Ningyi and Yu, Zihao and Ruixiao Zeng and Luo, Siqiang},
  booktitle={42nd International Conference on Machine Learning},
  year={2025},
  month={May},
  publisher={PMLR},
  volume={267},
  location={Vancouver, Canada},
  url={https://icml.cc/virtual/2025/poster/45740},
}
```

## Dependencies
### Python
Installed `env.txt` by conda:
```bash
conda create --name <env> --file env.txt
```

### C++
* C++ 14
* CMake 3.16
* [eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

## Experiment
### Data Preparation
1. Use `utils/data_transfer.py` to generate processed files under path `data/[dataset_name]` similar to the example folder `data/cora`:
  * `adj.npz`: adjacency matrix in scipy.sparse.csr_matrix
  * `feats.npy`: features in .npy array
  * `labels.npz`: node label information
    * 'label': labels (number or one-hot)
    * 'idx_train/idx_val/idx_test': indices of training/validation/test nodes
  * `adj_el.bin`, `adj_pl.bin`, `attribute.txt`, `degree.npz`: graph files for precomputation

### Decoupled Model Propagation
1. Compile Cython:
```bash
cd precompute
python setup.py build_ext --inplace
```

### Model Training
1. Run full-batch experiment: 
```bash
python run_fb.py -f [seed] -c [config_file] -v [device]
```
2. Run mini-batch experiment
```bash
python run_mb.py -f [seed] -c [config_file] -v [device]
```

## Reference & Links
### Datasets
* cora, citeseer, pubmed: [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
* arxiv, products, papers100m: [OGBl](https://ogb.stanford.edu/docs/home/)
* GenCAT: [GenCAT](https://github.com/seijimaekawa/GenCAT)

### Baselines
- [GLT](https://github.com/VITA-Group/Unified-LTH-GNN): *A Unified Lottery Ticket Hypothesis for Graph Neural Networks*
- [GEBT](https://github.com/GATECH-EIC/Early-Bird-GCN): *Early-Bird GCNs: Graph-Network Co-optimization towards More Efficient GCN Training and Inference via Drawing Early-Bird Lottery Tickets*
- [CGP](https://github.com/LiuChuang0059/CGP/): *Comprehensive Graph Gradual Pruning for Sparse Training in Graph Neural Networks*
- [DSpar](https://github.com/zirui-ray-liu/DSpar_tmlr): *DSpar: An Embarrassingly Simple Strategy for Efficient GNN Training and Inference via Degree-Based Sparsification*
- [NDLS](https://github.com/zwt233/NDLS): *Node Dependent Local Smoothing for Scalable Graph Learning*
- [NIGCN](https://github.com/kkhuang81/NIGCN): *Node-wise Diffusion for Scalable Graph Learning*

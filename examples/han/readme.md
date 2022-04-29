# Heterogeneous Graph Attention Network (HAN)

This is an implementation of `HAN` for heterogeneous graphs.
- Paper link: [https://arxiv.org/abs/1903.07293](https://arxiv.org/abs/1903.07293)
- Author's code repo: [https://github.com/Jhy1993/HAN](https://github.com/Jhy1993/HAN). Note that the original code is 
implemented with Tensorflow for the paper.

## Usage

`python han_trainer.py` for reproducing HAN's work on IMDB.
> Note: this scripts only support `IMDB`, which means command  `python han_trainer.py --dataset ACM` will not run on `ACM`.
> If you want to test the performance of other datasets, you are suggested to make some modification of the trainer script.

## Performance


Reference performance numbers for the IMDB dataset:
> train test val = 400, 3478, 400

| Dataset | Paper(80% train) | Our(tf)     |
| ---- |------------------|-------------|
| IMDB | 58.51            |  |


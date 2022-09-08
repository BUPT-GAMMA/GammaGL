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
(0.01, 200, 0.0001, 8, 0.8, 0.58178, 0.002811689883326394)
> train test val = 400, 3478, 400, about 9% for trianing

| Dataset | Paper(80% training) | Paper(60% training) | Paper(40% training) | Paper(20% training) | Our(tf) | Our(th) | Our(pd) |
| ------- | ------------------- | ------------------- | ------------------- | ------------------- | ------- | ------- | ------- |
| IMDB    | 58.51               | 58.32               | 57.97               | 55.73               | 58.17        |         |         |

```bash
TL_BACKEND=tensorflow python3 han_trainer.py --n_epoch 200 --lr 0.01 --l2_coef 0.0001 --heads 8 --drop_rate_1 0.8 --temp 0.9
```
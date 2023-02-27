# Heterogeneous Graph Propagation Network (HPN)

This is an implementation of `HPN` for heterogeneous graphs.

- Paper link: [https://ieeexplore.ieee.org/abstract/document/9428609](https://ieeexplore.ieee.org/abstract/document/9428609)

## Usage

`python hpn_trainer.py`

> Note: this scripts only support `IMDB`, which means command  `python hpn_trainer.py --dataset ACM` will not run on `ACM`.
> If you want to test the performance of other datasets, you are suggested to make some modification of the trainer script.

## Performance

Reference performance numbers for the IMDB dataset:
(0.01, 200, 0.0001, 8, 0.8, 0.58178, 0.002811689883326394)
> train test val = 400, 3478, 400, about 9% for trianing

| Dataset |   Our(tf)    |   Our(th)    |   Our(pd)    |
| ------- | ------------ | ------------ | ------------ |
| IMDB    | 58.05(±0.38) | 57.23(±0.47) | 57.75(±0.34) |

```bash
TL_BACKEND=tensorflow python3 hpn_trainer.py --lr 0.01 --hidden_dim 512 --iter_K 1 --l2_coef 0.001  --drop_rate 0.4 --alpha 0.3
TL_BACKEND=torch python3 hpn_trainer.py --lr 0.01 --hidden_dim 512 --iter_K 1 --l2_coef 0.001  --drop_rate 0.4 --alpha 0.3
TL_BACKEND=paddle python3 hpn_trainer.py --lr 0.01 --hidden_dim 512 --iter_K 1 --l2_coef 0.001  --drop_rate 0.4 --alpha 0.3
```
GIN

========================

- Paper link: [https://arxiv.org/abs/1810.00826](https://arxiv.org/abs/1810.00826)
- Author's code repo (in Pytorch):
  [https://github.com/weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns)

How to run
----------

Run with following:

```bash
# use tensorflow bakcend

TL_BACKEND=tensorflow python gin_trainer.py --dataset=MUTAG 
```

```bash
# use pytorch backend
TL_BACKEND=torch python gin_trainer.py --dataset=MUTAG 
```

```bash
# use paddle backend
TL_BACKEND=paddle python gin_trainer.py --dataset=MUTAG 

```

Results
-------

| Dataset | Paper      | Our(pd)    | Our(tf)    | Our(th)    |
|---------|------------|------------|------------|------------|
| MUTAG   | 89.4 ± 5.6 | 89.4 ± 5.6 | 89.4 ± 5.6 | 89.4 ± 5.6 |


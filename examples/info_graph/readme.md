InfoGraph 
========================

- Paper link: [https://arxiv.org/pdf/1908.01000](https://arxiv.org/pdf/1908.01000)
- Author's code repo (in Pytorch):
  [https://github.com/fanyun-sun/InfoGraph](https://github.com/fanyun-sun/InfoGraph)

  
How to run
----------

Run with following:

```bash
# use tensorflow background
TL_BACKEND=tensorflow python infograph_trainer.py --dataset=MUTAG  --hid_dim=32 --lr=0.01 --epochs=20 --n_layers=5
TL_BACKEND=tensorflow python infograph_trainer.py --dataset=IMDB-BINARY --hid_dim=32 --lr=0.01 --epochs=20 --n_layers=5
TL_BACKEND=tensorflow python infograph_trainer.py --dataset=REDDIT-BINARY --hid_dim=32 --lr=0.01 --epochs=20 --n_layers=5
```
```bash
# use pytorch background
TL_BACKEND=torch python infograph_trainer.py --dataset=MUTAG --hid_dim=32 --lr=0.01 --epochs=20 --n_layers=5
TL_BACKEND=torch python infograph_trainer.py --dataset=IMDB-BINARY --hid_dim=32 --lr=0.01 --epochs=20 --n_layers=5
TL_BACKEND=torch python infograph_trainer.py --dataset=REDDIT-BINARY --hid_dim=32 --lr=0.01 --epochs=20 --n_layers=5
```

Results
-------


|      Dataset      | MUTAG | IMDB-B | REDDIT-B |  
| :---------------: | :--:  | :----: |  :----:  |  
|   Author's Code   | 89.0  |  73.0  |   82.5   | 
|        DGL        | 89.88 |  72.7  |   88.5   | 
|     GammaGL(tf)   | 88.5  |  72.0  |   82.4   |  
|     GammaGL(th)   | --.-  |  --.-  |   --.-   |  
|     GammaGL(pd)   | --.-  |  --.-  |   --.-   |  
|     GammaGL(ms)   | --.-  |  --.-  |   --.-   |  
  


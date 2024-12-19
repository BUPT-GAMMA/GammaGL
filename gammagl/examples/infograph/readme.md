InfoGraph 
========================

- Paper link: [https://arxiv.org/pdf/1908.01000](https://arxiv.org/pdf/1908.01000)
- Author's code repo (in Pytorch):
  [https://github.com/fanyun-sun/InfoGraph](https://github.com/fanyun-sun/InfoGraph)

  
How to run
----------

Run with following:

```bash
# use tensorflow bakcend
TL_BACKEND="tensorflow" python infograph_trainer.py --dataset MUTAG 
TL_BACKEND="tensorflow" python infograph_trainer.py --dataset IMDB-BINARY
TL_BACKEND="tensorflow" python infograph_trainer.py --dataset REDDIT-BINARY
```
```bash
# use pytorch backend
TL_BACKEND="torch" python infograph_trainer.py --dataset MUTAG 
TL_BACKEND="torch" python infograph_trainer.py --dataset IMDB-BINARY
TL_BACKEND="torch" python infograph_trainer.py --dataset REDDIT-BINARY 
```
```bash
# use paddle backend
TL_BACKEND="paddle" python infograph_trainer.py --dataset MUTAG 
TL_BACKEND="paddle" python infograph_trainer.py --dataset IMDB-BINARY
TL_BACKEND="paddle" python infograph_trainer.py --dataset REDDIT-BINARY
```

Results (accuracy)
-------


|      Dataset      | MUTAG | IMDB-B | REDDIT-B |  
| :---------------: | :---:  | :----: |  :----:  |  
|   Author's Code   | 89.0  |  73.0  |   82.5   | 
|        DGL        | 89.88 |  72.7  |   88.5   | 
|     GammaGL(tf)   | 89.2 ± 0.25  |  72.0  |   82.4   |  
|     GammaGL(th)   | 90.65 ± 0.2 |  --.-  |   --.-   |  
|     GammaGL(pd)   | 89.2 ± 0.53 | 72.13 ± 0.12| 84.8 |  
|     GammaGL(ms)   | --.-  |  --.-  |   --.-   |  
  


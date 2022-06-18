SIGN: Scalable Inception Graph Neural Networks
===============
This repository contains the code to run the SIGN model on the Flickr dataset
- paper link: [https://arxiv.org/pdf/2004.11198.pdf](https://arxiv.org/pdf/2004.11198.pdf)

Results
---------------
### Flickr

```bash
# use TensorFlow background
TL_BACKEND=tensorflow python sign_trainer.py
# use Paddle background
TL_BACKEND=paddle python sign_trainer.py
# use Pytorch background
TL_BACKEND=torch python sign_trainer.py
```


|      Dataset      | Flickr | 
| :---------------: | :--: |
|   Author's Code   | 51.4 |
|     GammaGL(tf)   | 51.9 |
|     GammaGL(th)   | 51.9 |
|     GammaGL(pd)   | 51.9 |
|     GammaGL(ms)   | --.- |




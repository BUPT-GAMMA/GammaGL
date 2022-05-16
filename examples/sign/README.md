SIGN: Scalable Inception Graph Neural Networks
===============
This repository contains the code to run the SIGN model on the Flickr dataset
If you want to know more about SIGN checkout its arXiv page and its ICML 2020 GRL+ Workshop version.
- paper link: [https://arxiv.org/pdf/2004.11198.pdf](https://arxiv.org/pdf/2004.11198.pdf)

Results
---------------
### Flickr

```bash
# use TensorFlow background
CUDA_VISIBLE_DEVICES="1" TL_BACKEND="tensorflow" python sign_trainer.py
# use Paddle background
CUDA_VISIBLE_DEVICES="1" TL_BACKEND="paddle" python sign_trainer.py
# use Pytorch background
CUDA_VISIBLE_DEVICES="1" TL_BACKEND="torch" python sign_trainer.py
```


|      Dataset      | Flickr | 
| :---------------: | :--: |
|   Author's Code   | 51.4 |
|     GammaGL(tf)   | 51.9 |
|     GammaGL(th)   | 51.9 |
|     GammaGL(pd)   | 51.9 |
|     GammaGL(ms)   | --.- |




# Heterogeneous Graph Attention Network (HGAT)

This is an implementation of `HAN` for heterogeneous graphs.

- Paper link: [https://aclanthology.org/D19-1488/](https://aclanthology.org/D19-1488/)
- Author's code repo: [https://github.com/BUPT-GAMMA/HGAT](https://github.com/BUPT-GAMMA/HGAT). Note that the original code is 
  implemented with Tensorflow for the paper.

## Usage

`python hgat_trainer.py` for reproducing HGAT's work on IMDB.



## Performance



| Dataset  |Paper(80% training)  | Our(tf)      | Our(th)      | Our(pd)      |
| -------  | ------------------  | -------      | -------      |--------      |
| AGNews   |  72.10             |    63.80      |              |              |
| Ohsumed  |   42.68            | 25.82         |              |              |
| Twitter  |     63.21          |   61.06       |              |              |
| IMDB     |                    |     57.71     |              |              |

```bash
TL_BACKEND="tensorflow" python3 hgat_trainer.py --n_epoch 100 --lr 0.01 --l2_coef 0.0001 --drop_rate 0.8

```
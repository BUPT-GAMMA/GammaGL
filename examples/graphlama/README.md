# GraphLAMA: Enabling Efficient Adaptation of Graph Language Models with Limited Annotations
* Paper link: https://arxiv.org/pdf/2506.21559
* Author's code repo: https://github.com/BUPT-GAMMA/GraphLAMA.git

# How to Run

* First, follow the [GraphLAMA original repo](https://github.com/BUPT-GAMMA/GraphLAMA.git) to install all required packages where will tell you to follow the instructions to 

* Then download all required datasets and pretrained checkpoints, and fill their path into corresponding values in eval.sh

# Dataset Statics
| Dataset | # Nodes | # Edges |  # Classes | 
| :-------: | :-------: | :------: | :------: |
| Cora | 25,120 | 182,280 | 70 |
| PubMed | 19,717 | 44,338 | 3 |
| ogb-arxiv | 169,343 | 1,166,243 | 40 |

# Files Description
* graphgpt_trainer.py: the trainer of graphgpt, inference stage
* graphgpt_eval.py: run this to evaluate 

# Results
```bash
# run inference
TL_BACKEND="torch" nohup bash examples/graphgpt/eval.sh > log/test_graphgpt.out &
# run evaluation
python examples/graphgpt/graphgpt_eval.py --dataset cora
```
| Dataset | Paper | Our(torch) |
| :-------: | :-------: | :------: |
| Cora | 0.1501 | 0.1451 | 
# LLaGA: Large Language and Graph Assistant
* Paper link: http://arxiv.org/abs/2402.08170
* Author's code repo: https://github.com/VITA-Group/LLaGA

# How to Run

* First, follow the original repo to install all required packages;

* Then download all required datasets and pretrained checkpoints, and fill their path into corresponding values in eval.sh

# Dataset Statics
| Dataset | # Nodes | # Edges |  # Classes | 
| :-------: | :-------: | :------: | :------: |
| Cora | 2,708 | 5,429 | 7 |
| PubMed | 19,717 | 44,338 | 3 |
| Arxiv | 169,343 | 1,166,243 | 40 |

# Files Description
* llaga_trainer.py: the trainer of graphgpt, inference stage
* llaga_eval.py: run this to evaluate 

# Results
```bash
# run inference
TL_BACKEND="torch" nohup bash examples/llaga/eval.sh > log/test_llaga.out &
# run evaluation
python examples/llaga/llaga_eval.py --dataset cora --task nc --res_path examples/llaga/test.txt # "output_path" you specified in eval.sh
```
| Dataset | Paper | Our(torch) |
| :-------: | :-------: | :------: |
| Cora | 0.8782 | 0.8727 | 
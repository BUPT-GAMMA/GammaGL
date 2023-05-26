# SpCo
This repo is for source code of NeurIPS 2022 paper "Revisiting Graph Contrastive Learning from the Perspective of Graph Spectrum". \
Paper Link: https://arxiv.org/abs/2210.02330

# Environment Settings
```
python==3.8.5
gammagl==0.1.0
networkx==2.5
numpy==1.19.5
scikit_learn==1.0
scipy==1.9.1
torch==1.7.0
tensorlayerx==0.5.8
```
GPU: GeForce RTX 3090 \
CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

# Usage
First, go into the target folder. Then, run the following commands:
```
# DGI+SpCo
python execute.py cora --gpu=0

# GRACE+SpCo
python train.py cora --gpu_id=0

# CCA+SpCo
python main.py cora --gpu 0
```
where "cora" can be replaced by {citeseer, blog, flickr, pubmed}. \
For each target model, we just add our SpCo on original code with some adaption. Therefore, you can refer to original code for better understanding about our code.

# Cite

# Contact
If you have any questions, please feel free to contact me with {nianliu@bupt.edu.cn}

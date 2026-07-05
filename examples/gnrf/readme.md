# Graph Neural Ricci Flow: Evolving Feature From A Curvature Perspective
---

- Paper link: https://proceedings.iclr.cc/paper_files/paper/2025/file/4d3ac0eee841e6df6e08e51932943266-Paper-Conference.pdf
- Author's code repo (in PyTorch): https://github.com/GalenChen320/GNRF_new

## Datasets and Performances
---

|Datasets|Cornell|Wisconsin|Texas|Roman-Empire|Tolokers|Cora_Full|Pubmed|
|---|---|---|---|---|---|---|---|
|Hom.level|0.1227|0.1778|0.0609|0.0000|0.6344|0.5670|0.8024|
|#Node|183|251|183|22,662|11,758|19,793|19,717|
|Paper|87.28(±3.12)|88.00(±2.00)|87.39(±4.13)|86.25(±0.46)|83.96(±0.39)|72.12(±0.50)|90.37(±0.69)|
|Ours|79.46(±5.57)|87.60(±2.33)|84.86(±6.64)|85.01(±1.04)|81.14(±0.98)|68.62(±0.59)|88.85(±0.39)|

|Datasets|Ogbn-Arxiv|
|---|---|
|depth|3|
|num-hid|64|
|Paper|69.33|
|Ours|60.01|

## Notes
---

- On the Cornell dataset, under the source code and environment described in the paper, the performance on an RTX 3090 is mean: 79.46, std: 7.77. Therefore, the GammaGL version retains mean: 79.46, std: 5.57.
- For the Ogbn-arxiv dataset, with depth=3, num-hid=64, no standard deviation data was reported in the paper. In the original paper's source code environment on an RTX 3090, the results are mean: 66.64, std: 0.62, while the GammaGL version retains mean: 60.01, std: 1.91.
- When using PaddlePaddle or MindSpore as backends, due to the lack of mature and unified Neural ODE solving ecosystems, the odeint module is manually implemented, with only correctness testing performed and no performance guarantees.
- All the data presented in the above tables are obtained only with the PyTorch backend.

## How To Run
---

Execute in the current directory.


```python
python train.py --dataset wisconsin
```

The dataset defaults to GPU mode; under CPU, the command is as follows. Note that the specification of CPU for different backends is case-sensitive.


```python
python train.py --dataset wisconsin  --device cpu
```

# EdgePrompt / EdgePrompt+

- Paper link: [https://arxiv.org/abs/2503.00750](https://arxiv.org/abs/2503.00750)
- Author's code repo: [https://github.com/xbfu/EdgePrompt](https://github.com/xbfu/EdgePrompt)

## Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| PubMed   | 19,717  | 88,651  | 3         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

### Optional pretraining

```bash
cd examples/edgeprompt
TL_BACKEND=torch python node_edgeprompt_pretrain.py --dataset Cora --epochs 100 --seed 0
```

### Downstream EdgePrompt

```bash
cd examples/edgeprompt
TL_BACKEND=torch python node_edgeprompt_finetune.py --dataset Cora --method edgeprompt --num_shots 5 --pretrained_path ./cora_ep_gppt_backbone.npz --epochs 100 --seed 0
```

### Downstream EdgePrompt+

```bash
cd examples/edgeprompt
TL_BACKEND=torch python node_edgeprompt_finetune.py --dataset Cora --method edgeprompt_plus --num_shots 5 --pretrained_path ./cora_ep_gppt_backbone.npz --epochs 100 --seed 0
```

## Results

| Dataset  | Method       | Paper | Our(Torch) |
|----------|--------------|-------|------------|
| Cora     | EdgePrompt   | 37.26 | 73.20      |
| Cora     | EdgePrompt+  | 56.41 | 73.94      |
| CiteSeer | EdgePrompt   | 29.83 | 46.77      |
| CiteSeer | EdgePrompt+  | 43.49 | 47.37      |
| PubMed   | EdgePrompt   | 47.20 | 51.05      |
| PubMed   | EdgePrompt+  | 61.51 | 55.62      |

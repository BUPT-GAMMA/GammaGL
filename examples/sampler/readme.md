# Graph Neighbor Sampler in Gammagl

# Dataset Statics

|   Dataset   |    Nodes     | Edges |
|:-----------:|:------------:|:-----:|
|   Reddit    |   232965      | 114615892 |

# Some experiments in CPU sampler

subgraph sampler with sample lists [25,10] and batch_size 1024.

```bash
python neighbor_sampler.py --sample_lists=25,10 --batch_size=1024
```

subgraph sampler with one-layer full neighbors and batch_size 2048.

```bash
python neighbor_sampler.py --sample_lists=-1 --batch_size=2048
```

# Some experiments in GPU sampler

subgraph sampler with sample lists [25,10] and batch_size 1024 in tensorflow backend.

```bash
python neighbor_sampler_gpu.py --sample_lists=25,10 --batch_size 1024 --backend=tensorflow
```

subgraph sampler with one-layer full neighbors and batch_size 1024 in torch backend.

```bash
python neighbor_sampler_gpu.py --sample_lists=-1 --batch_size=1024 --backend=torch
```

subgraph sampler with one-layer full neighbors and batch_size 2048 in paddle backend.

```bash
python neighbor_sampler_gpu.py --sample_lists=-1 --batch_size=2048 --backend=paddle
```

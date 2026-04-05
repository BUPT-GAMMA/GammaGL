Adaptive Message Passing (AMP)
====================================


- Paper link:[https://arxiv.org/pdf/2312.16560]
- Author's code repo:[https://github.com/nec-research/Adaptive-Message-Passing]

What this example does
----------------------


How to run
----------



```bash
cd examples/amp
python examples/amp/amp_trainer.py \
  --dataset_path examples/amp \
  --zinc_local_zip examples/amp/ZINC/molecules.zip \
  --gpu 0
```




Results
-------


| Dataset| paper | our |
| :-----: | :-----: | :------: |
| Diameter |  63% |  68%    |
|SSSP| 72%   |  68% |
|Eccentricity| 32% | 28% |
|ZINC|0.7065±0.0105|0.6943±0.0047|
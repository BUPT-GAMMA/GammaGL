# Simple and Deep Graph Convolutional Networks (GCNII)

- Paper link: [https://arxiv.org/abs/2007.02133](https://arxiv.org/abs/2007.02133)
- Author's code repo: [https://github.com/chennnM/GCNII](https://github.com/chennnM/GCNII). 
> Note that our implementation is little different with the author's in the optimizer.  The author applied different weight decay coefficient on learnable paramenters, while TensorLayerX has not support this feature.


Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gcnii_train.py --dataset cora
```


| Dataset  | Paper | Our(pd)    | Our(tf) |
|----------|-------|-------------|---------|
| cora     | 85.5 | 83.12(0.47) |83.23(0.76)|
| pubmed   | 73.4 | 72.04(0.91) |71.9(0.7)|
| citeseer | 80.3 | 80.36(0.65) |80.1(0.5)|


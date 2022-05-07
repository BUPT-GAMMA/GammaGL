Graph Attention Networks v2 (GATv2)
============

- Paper link: [How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf)
- Author's code repo: [https://github.com/tech-srl/how_attentive_are_gats](https://github.com/tech-srl/how_attentive_are_gats).
- Annotated implemetnation: [https://nn.labml.ai/graphs/gatv2/index.html]


Run with following:

```bash
python3 gatv2_trainer.py
```

Results
-------

| Dataset  | Our(pd)     | Our(tf)     |
| -------- | ----------- | ----------- |
| cora     | 82.45(0.34) | 81.78(0.29) |
| pubmed   | 70.9(1.28)  | 69.9(0.23)  |
| citeseer | 78.46(0.19) | 77.49(0.08) |

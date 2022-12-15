# Dynamic Graph CNN(DGCNN)

- paper link:https://arxiv.org/pdf/1801.07829.pdf
- Author's code repo:[WangYueFt/dgcnn (github.com)](https://github.com/WangYueFt/dgcnn). The code has both included pytorch & tensorflow versions.

# Dataset statics

- Dataset:ModelNet40
- 12,311 meshed CAD models
- 40 categories
- 9,843 models are used for training and 2,468 models are for testing. 

refer to [ModelNet40](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.ModelNet40)

# results

|  Backend   |      accuracy      |    avg accuracy    |
|:----------:|:------------------:|:------------------:|
| tensorflow | 0.8893841166936791 | 0.8394360465116278 |
|   torch    | 0.880064829821718  | 0.8215581395348837 |
|   paddle   |                    |                    |


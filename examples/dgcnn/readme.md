# Dynamic Graph CNN(DGCNN)

- paper link:https://arxiv.org/pdf/1801.07829.pdf
- Author's code repo:[WangYueFt/dgcnn (github.com)](https://github.com/WangYueFt/dgcnn). The code has both included pytorch & tensorflow versions.

# Dataset statics

- Dataset:ModelNet40
- 12,311 meshed CAD models
- 40 categories
- 9,843 models are used for training and 2,468 models are for testing. 

refer to [ModelNet40](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.ModelNet40)

# Results

|  Dataset   | Paper | Paper(avg) | Our(tf) | Our(tf,avg) | Our(th) | Our(th,avg) |
|:----------:|:-----:|:----------:|:-------:|:-----------:|:-------:|:-----------:|
| ModelNet40 | 91.7  |    88.9    |  88.94  |    83.94    |  88.01  |    82.16    |

Notes: accuracy with 'avg' refers to 'average accuracy', which calls the method of 
[balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html); accuracy without 'avg' refers to 'overall accuracy', which calls the method of [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).

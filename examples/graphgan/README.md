# Graph Generative Adversarial Nets (GraphGAN)
* Paper link: https://arxiv.org/pdf/1711.08267.pdf
* Author's code repo: https://github.com/hwwang55/GraphGAN. Note that the original code is implemented with Tensorflow for the paper.
# Dataset Statics
| Dataset | # Nodes | # Edges | 
| :-------: | :-------: | :------: | 
| arXiv-GrQc | 5242 | 14496 |
# Files Description
* gan_datasets : folds to save training data
  * CA-GrQc_train : training data
  * CA-GrQc_test : positive sampling of test data 
  * CA-GrQc_test_neg : negative sampling of test data
  > The data should be an undirected graph in which node IDs start from 0 to N-1 (N is the number of nodes in the graph). Each line contains two node IDs indicating an edge in the graph.

  > txt file sample :
  > 0  1
  > 3  2
  > . . .

  * CA-GrQc_pre_train.emb : pre-trained node embeddings
* gan_cache : folds to save constructed BFS-trees
  * CA-GrQc.pkl : save constructed BFS-trees
* gan_results
  * CA-GrQc : evaluation results
  * CA-GrQc_gen_.emb : the learned embeddings of the generator with the best training result
  * CA-GrQc_dis_.emb : the learned embeddings of the discriminator with the best training result
  >Note: the dimension of pre-trained node embeddings should equal n_emb in src/GraphGAN/config.py
* checkpoint : folds to save the model weights
# Results
```
python graphgan_trainer.py --batch_size_dis 1024 --batch_size_gen 1024 --n_epochs 30 --lr_gen 1e-5 --lr_dis 1e-5
```
| Dataset | Paper | Our(pd) | Our(tf) | 
| :-------: | :-------: | :------: | :------: | 
| arXiv-GrQc | 84.9 | ... | ... |
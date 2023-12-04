# Spectral Graph Neural Networks Meet Transformers (SpecFormer)

*   Paper link: <http://www.shichuan.org/doc/148.pdf>
*   Author's code repo: <https://github.com/BUPT-GAMMA/Specformer>. Note that the original code is implemented with Torch for the paper.

# Dataset Statics

| Dataset | # Nodes | # Edges | # Classes |
| ------- | ------- | ------- | --------- |
| Chameleon    | 2,277   | 36,101  | 5         |
| Squirrel    | 5,201   | 217,073  | 5         |
| Cora    | 2,708   | 10,556  | 7         |


Refer to [WikipediaNetwork](https://github.com/BUPT-GAMMA/GammaGL/blob/main/gammagl/datasets/wikipedia_network.py),[Planetoid](https://github.com/BUPT-GAMMA/GammaGL/blob/main/gammagl/datasets/planetoid.py)  


## Results

```bash
# available dataset: "chameleon", "squirrel", config below
python spec_trainer.py --dataset=chameleon --n_epoch=500 --n_heads=4 --n_layer=2 --hidden_dim=32 --lr=0.001 --weight_decay=0.0005 --tran_dropout=0.2 --feat_dropout=0.4 --prop_dropout=0.5 
python spec_trainer.py --dataset=squirrel --n_epoch=500 --n_heads=4 --n_layer=2 --hidden_dim=32 --lr=0.001 --weight_decay=0.001 --tran_dropout=0.1 --feat_dropout=0.4 --prop_dropout=0.4 
python spec_trainer.py --dataset=cora --n_epoch=500 --n_heads=4 --n_layer=2 --hidden_dim=32 --lr=0.0002 --weight_decay=0.0001 --tran_dropout=0.2 --feat_dropout=0.6 --prop_dropout=0.2 


```

| Dataset     | Paper        | Our(th)    |
| -------     | -----        | ---------- |
| chameleon   | 74.72±1.29   | 76.29±4.54 |
| squirrel    |64.64±0.81    | 65.31±1.55 |
| cora        |88.57±1.01    | 87.26±0.01  |


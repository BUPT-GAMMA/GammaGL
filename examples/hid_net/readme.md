Representation Learning on Graphs with Jumping Knowledge Networks (JK-Net)
============

- Paper link: [https://arxiv.org/abs/2312.08616](https://arxiv.org/abs/2312.08616)
- Author's code repo: [https://github.com/BUPT-GAMMA/HiD-Net](https://github.com/BUPT-GAMMA/HiD-Net). Note that the original code is 
implemented with Torch for the paper. 


How to run
----------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python hid_trainer.py --dataset cora 
```


Results
-------
```bash
TL_BACKEND="mindspore" python hid_trainer.py --dataset cora --alpha 0.1 --beta 0.9 --gamma 0.3 --k 10 --hidden 128 --lr 0.01 --weight_decay 0 --dropout 0.55
TL_BACKEND="mindspore" python hid_trainer.py --dataset citeseer --alpha 0.1 --beta 0.9 --gamma 0.2 --k 10 --hidden 64 --lr 0.005 --weight_decay 0.05 --dropout 0.5
TL_BACKEND="mindspore" python hid_trainer.py --dataset pubmed --alpha 0.08 --beta 0.92 --gamma 0.3 --k 8 --hidden 32 --lr 0.02 --weight_decay 0.0005 --dropout 0.5
TL_BACKEND="paddle" python hid_trainer.py --dataset cora --alpha 0.1 --beta 0.9 --gamma 0.3 --k 10 --hidden 128 --lr 0.01 --weight_decay 0 --dropout 0.55
TL_BACKEND="paddle" python hid_trainer.py --dataset citeseer --alpha 0.1 --beta 0.9 --gamma 0.2 --k 10 --hidden 64 --lr 0.005 --weight_decay 0.05 --dropout 0.5
TL_BACKEND="paddle" python hid_trainer.py --dataset pubmed --alpha 0.08 --beta 0.92 --gamma 0.3 --k 8 --hidden 32 --lr 0.02 --weight_decay 0.0005 --dropout 0.5
TL_BACKEND="tensorflow" python hid_trainer.py --dataset cora --alpha 0.1 --beta 0.9 --gamma 0.3 --k 10 --hidden 128 --lr 0.01 --weight_decay 0 --dropout 0.55
TL_BACKEND="tensorflow" python hid_trainer.py --dataset citeseer --alpha 0.1 --beta 0.9 --gamma 0.2 --k 10 --hidden 64 --lr 0.005 --weight_decay 0.05 --dropout 0.5
TL_BACKEND="tensorflow" python hid_trainer.py --dataset pubmed --alpha 0.08 --beta 0.92 --gamma 0.3 --k 8 --hidden 32 --lr 0.02 --weight_decay 0.0005 --dropout 0.5
TL_BACKEND="torch" python hid_trainer.py --dataset cora  --alpha 0.1 --beta 0.9 --gamma 0.3 --k 10 --hidden 128 --lr 0.01 --weight_decay 0 --dropout 0.55 
TL_BACKEND="torch" python hid_trainer.py --dataset citeseer --alpha 0.1 --beta 0.9 --gamma 0.2 --k 10 --hidden 64 --lr 0.005 --weight_decay 0.05 --dropout 0.5 
TL_BACKEND="torch" python hid_trainer.py --dataset pubmed --alpha 0.08 --beta 0.92 --gamma 0.3 --k 8 --hidden 32 --lr 0.02 --weight_decay 0.0005 --dropout 0.5
```
| Dataset | Paper | Our(ms) | Our(pd) | Our(tf) | Our(th) |
| ---- | ---- | --- | ---- | ---- | ---- |
| cora | 0.840(±0.6) | 0.8078(±0.0049) | 0.8274(±0.0190) | 0.8218(±0.0024) | 0.8138(±0.00024) |
| citeseer | 0.732(±0.2) | 0.7138(±0.0019) | 0.7134(±0.0012) | 0.7140(±0.0022)| 0.7134(±0.0022)|
| pubmed | 0.811(±0.1) | 0.7996(±0.0030) | 0.7910(±0.0044) | 0.8026(±0.0034) | 0.7938(±0.0151) | 0.7920(±0.0097)| 
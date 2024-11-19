# FedHGNN
The source code of WWW 2024 paper "Federated Heterogeneous Graph Neural Network for Privacy-preserving Recommendation"


# Requirements
```
dgl==1.1.0+cu113
numpy==1.21.6
ogb==1.3.6
python==3.7.13
scikit-learn==1.0.2
scipy==1.7.3
torch==1.12.1+cu113
torchaudio==0.12.1+cu113
torchvision==0.13.1+cu113
```


# Easy Run
```
cd ./codes/FedHGNN
python main.py --dataset acm --shared_num 20 --p1 1 --p2 2 --lr 0.01 --device cuda:0
```



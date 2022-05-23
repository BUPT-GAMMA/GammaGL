#! /bin/sh
#

# Below is for homophily datasets, sparse split

for i in 1 2 3 4 5 6 7 8 9 10
do
    python gprgnn_trainer.py --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset cora \
        --lr 0.01 \
        --alpha 0.1
done
python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do
    python gprgnn_trainer.py --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset citeseer \
        --lr 0.01 \
        --alpha 0.1        
done
python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do
    python gprgnn_trainer.py --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset pubmed \
        --lr 0.05 \
        --alpha 0.2 
done
python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do        
    python gprgnn_trainer.py --train_rate 0.025 \
            --val_rate 0.025 \
            --dataset computers \
            --lr 0.05 \
            --alpha 0.5
done
# python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do        
    python gprgnn_trainer.py --train_rate 0.025 \
            --val_rate 0.025 \
            --dataset photo \
            --lr 0.01 \
            --alpha 0.5
done
python process.py

# Below is for heterophily datasets, dense split
for i in 1 2 3 4 5 6 7 8 9 10
do
    python gprgnn_trainer.py --train_rate 0.6 \
            --val_rate 0.2 \
            --dataset chameleon \
            --lr 0.05 \
            --alpha 1.0 \
            --l2_coef 0.0 \
            --dprate 0.7 
done
python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do
    python gprgnn_trainer.py --train_rate 0.6 \
            --val_rate 0.2 \
            --dataset squirrel \
            --lr 0.05 \
            --alpha 0.0 \
            --dprate 0.7 \
            --l2_coef 0.0
done
python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do
    python gprgnn_trainer.py --train_rate 0.6 \
            --val_rate 0.2 \
            --dataset texas \
            --lr 0.05 \
            --alpha 1.0
done
python process.py

for i in 1 2 3 4 5 6 7 8 9 10
do       
    python gprgnn_trainer.py --train_rate 0.6 \
            --val_rate 0.2 \
            --dataset cornell \
            --lr 0.05 \
            --alpha 0.9 
done
python process.py
#!/bin/bash
source /home/llh/miniconda3/etc/profile.d/conda.sh
conda activate ggl
cd /data/zhaxi/GammaGL/examples/defog/
TL_BACKEND=torch python defog_trainer.py \
  --dataset tree --data_root ./datasets \
  --save_dir ./checkpoints_tree_seed0_final --seed 0 --gpu 1 \
  --n_layers 10 --hidden_mlp_X 128 --hidden_mlp_E 64 --hidden_mlp_y 128 \
  --dx 256 --de 64 --dy 64 --dim_ffX 256 --dim_ffE 64 --dim_ffy 256 \
  --n_head 8 --n_epochs 100000 --batch_size 64 --lr 2e-4 \
  --train_distortion polydec --sample_distortion polydec \
  --omega 0 --eta 0 --sample_steps 1000 \
  --check_val_every_n_epochs 2000 --sample_every_val 1 --val_num_samples 40

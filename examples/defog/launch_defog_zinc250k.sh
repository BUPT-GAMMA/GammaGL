#!/bin/bash
source /home/llh/miniconda3/etc/profile.d/conda.sh
conda activate ggl
cd /data/zhaxi/GammaGL/examples/defog/
TL_BACKEND=torch python defog_trainer.py \
  --dataset zinc250k \
  --data_root ./datasets \
  --save_dir ./checkpoints_zinc250k_seed0_final \
  --seed 0 \
  --gpu 7 \
  --n_layers 12 \
  --hidden_mlp_X 256 --hidden_mlp_E 128 --hidden_mlp_y 256 \
  --dx 256 --de 64 --dy 128 \
  --dim_ffX 256 --dim_ffE 128 --dim_ffy 256 \
  --n_head 8 \
  --rrwp_steps 20 \
  --n_epochs 300 \
  --batch_size 256 \
  --lr 2e-4 \
  --train_distortion polydec \
  --sample_distortion polydec \
  --omega 0.1 \
  --eta 300.0 \
  --check_val_every_n_epochs 4 \
  --sample_every_val 2 \
  --val_num_samples 256

#!/bin/bash
source /home/llh/miniconda3/etc/profile.d/conda.sh
conda activate ggl
cd /data/zhaxi/GammaGL/examples/defog/
TL_BACKEND=torch python defog_trainer.py \
  --dataset qm9 --data_root ./datasets \
  --save_dir ./checkpoints_qm9_noh_seed0_final --seed 0 --gpu 5 \
  --n_layers 9 --n_epochs 1000 --batch_size 1024 --lr 2e-4 \
  --train_distortion identity --sample_distortion polydec \
  --sample_steps 500 --omega 0 --eta 0 \
  --check_val_every_n_epochs 50 --sample_every_val 1 --val_num_samples 512 \
  --remove_h

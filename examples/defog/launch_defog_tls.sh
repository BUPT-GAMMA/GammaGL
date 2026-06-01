#!/bin/bash
source /home/llh/miniconda3/etc/profile.d/conda.sh
conda activate ggl
cd /data/zhaxi/GammaGL/examples/defog/
TL_BACKEND=torch python defog_trainer.py \
  --dataset tls \
  --data_root ./datasets \
  --save_dir ./checkpoints_tls_seed0_final \
  --seed 0 \
  --gpu 6 \
  --resume_from ./checkpoints_tls_seed0_final \
  --start_epoch 14855

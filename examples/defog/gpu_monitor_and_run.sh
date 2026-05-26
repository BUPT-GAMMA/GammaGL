#!/bin/bash
# GPU Monitor: watches for free GPUs and launches DeFoG tasks
# Priority: 1) planar sampling  2) sbm training
# State files to track progress
SCRIPT_DIR="/data/zhaxi/GammaGL/examples/defog"
STATE_DIR="$SCRIPT_DIR/.monitor_state"
mkdir -p "$STATE_DIR"

PLANAR_DONE="$STATE_DIR/planar_sampling_done"
SBM_DONE="$STATE_DIR/sbm_training_done"
LOG_DIR="$SCRIPT_DIR/logs"

# Threshold: GPU is considered "free" if memory usage < 10% (8GB on 80GB card)
FREE_THRESHOLD_GB=8

check_free_gpu() {
    # Returns the first free GPU id, or -1 if none free
    for gpu_id in 0 1 2 3 4 5 6 7; do
        used_mb=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$used_mb" ]; then
            used_gb=$((used_mb / 1024))
            if [ "$used_gb" -lt "$FREE_THRESHOLD_GB" ]; then
                echo $gpu_id
                return 0
            fi
        fi
    done
    echo -1
    return 1
}

launch_planar_sampling() {
    local gpu_id=$1
    echo "[$(date)] Found free GPU $gpu_id, launching planar sampling..."

    cd "$SCRIPT_DIR"
    nohup /data/llh_conda_envs/ggl/bin/python defog_trainer.py \
        --dataset planar \
        --n_layers 10 \
        --hidden_mlp_X 128 \
        --hidden_mlp_E 64 \
        --hidden_mlp_y 128 \
        --dx 256 --de 64 --dy 64 \
        --n_head 8 \
        --dim_ffX 256 --dim_ffE 64 --dim_ffy 256 \
        --ema_decay 0.999 \
        --transition marginal \
        --extra_features rrwp \
        --rrwp_steps 12 \
        --sample \
        --evaluate \
        --sample_steps 1000 \
        --sample_distortion polydec \
        --omega 0.05 \
        --eta 50 \
        --num_samples 512 \
        --num_sample_fold 3 \
        --rdb general \
        --rdb_crit max_marginal \
        --gpu $gpu_id \
        --save_dir ./checkpoints_planar \
        --batch_size 64 \
        > "$LOG_DIR/planar_sampling.log" 2>&1 &

    echo $! > "$STATE_DIR/planar_sampling_pid"
    touch "$PLANAR_DONE"
    echo "[$(date)] Planar sampling launched on GPU $gpu_id, PID saved."
}

launch_sbm_training() {
    local gpu_id=$1
    echo "[$(date)] Found free GPU $gpu_id, launching sbm training..."

    cd "$SCRIPT_DIR"
    nohup /data/llh_conda_envs/ggl/bin/python defog_trainer.py \
        --dataset sbm \
        --n_layers 8 \
        --hidden_mlp_X 128 \
        --hidden_mlp_E 64 \
        --hidden_mlp_y 128 \
        --dx 256 --de 64 --dy 64 \
        --n_head 8 \
        --dim_ffX 256 --dim_ffE 64 --dim_ffy 256 \
        --ema_decay 0.999 \
        --transition absorbfirst \
        --extra_features rrwp \
        --rrwp_steps 20 \
        --n_epochs 50000 \
        --batch_size 32 \
        --lr 2e-4 \
        --rdb general \
        --rdb_crit max_marginal \
        --gpu $gpu_id \
        --save_dir ./checkpoints_sbm \
        --sample_steps 1000 \
        --num_samples 512 \
        --num_sample_fold 3 \
        > "$LOG_DIR/sbm_train.log" 2>&1 &

    echo $! > "$STATE_DIR/sbm_training_pid"
    touch "$SBM_DONE"
    echo "[$(date)] SBM training launched on GPU $gpu_id, PID saved."
}

# Check if planar sampling already running or done
if [ -f "$STATE_DIR/planar_sampling_pid" ]; then
    pid=$(cat "$STATE_DIR/planar_sampling_pid")
    if kill -0 "$pid" 2>/dev/null; then
        echo "[$(date)] Planar sampling already running (PID $pid), skipping."
    else
        rm -f "$STATE_DIR/planar_sampling_pid"
        # Check if it completed successfully
        if grep -q "Done!" "$LOG_DIR/planar_sampling.log" 2>/dev/null; then
            echo "[$(date)] Planar sampling already completed successfully."
        else
            echo "[$(date)] Planar sampling process died, resetting state to retry."
            rm -f "$PLANAR_DONE"
        fi
    fi
fi

if [ -f "$STATE_DIR/sbm_training_pid" ]; then
    pid=$(cat "$STATE_DIR/sbm_training_pid")
    if kill -0 "$pid" 2>/dev/null; then
        echo "[$(date)] SBM training already running (PID $pid), skipping."
    else
        rm -f "$STATE_DIR/sbm_training_pid"
        if grep -q "Done!" "$LOG_DIR/sbm_train.log" 2>/dev/null || grep -q "Training complete" "$LOG_DIR/sbm_train.log" 2>/dev/null; then
            echo "[$(date)] SBM training already completed successfully."
        else
            echo "[$(date)] SBM training process died, resetting state to retry."
            rm -f "$SBM_DONE"
        fi
    fi
fi

# Also check if sbm completed during training (includes sampling+eval)
if grep -q "Done!" "$LOG_DIR/sbm_train.log" 2>/dev/null; then
    touch "$SBM_DONE"
fi

echo "[$(date)] Checking GPU availability..."
free_gpu=$(check_free_gpu)

if [ "$free_gpu" -eq -1 ]; then
    echo "[$(date)] No free GPU found. Will check again later."
else
    echo "[$(date)] Free GPU found: $free_gpu"

    # Priority 1: planar sampling
    if [ ! -f "$PLANAR_DONE" ]; then
        launch_planar_sampling $free_gpu
    # Priority 2: sbm training
    elif [ ! -f "$SBM_DONE" ]; then
        launch_sbm_training $free_gpu
    else
        echo "[$(date)] All tasks completed! Nothing to do."
    fi
fi

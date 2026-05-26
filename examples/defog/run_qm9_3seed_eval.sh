#!/bin/bash
# QM9 3-seed 验证脚本
# 用法: ./run_qm9_3seed_eval.sh

cd /home/zxjc/GammaGL/examples/defog

# 定义任务队列
# 格式: "GPU 版本 seed 检查点目录"
declare -a TASKS=(
    "2 noh 0 checkpoints_qm9_noh_fixval"
    "3 withh 0 checkpoints_qm9_withh_v2"
    "2 noh 1 checkpoints_qm9_noh_fixval"
    "3 withh 1 checkpoints_qm9_withh_v2"
    "2 noh 2 checkpoints_qm9_noh_fixval"
    "3 withh 2 checkpoints_qm9_withh_v2"
)

LOG_DIR="./logs/qm9_3seed_eval"
mkdir -p "$LOG_DIR"

# 记录开始时间
echo "=== QM9 3-Seed Evaluation Started at $(date '+%Y-%m-%d %H:%M:%S') ===" | tee "$LOG_DIR/master.log"

# 按顺序执行任务（2卡并行，每轮2个任务）
for i in "${!TASKS[@]}"; do
    IFS=' ' read -r GPU VERSION SEED CKPT <<< "${TASKS[$i]}"

    # 构建参数
    if [ "$VERSION" = "noh" ]; then
        H_FLAG="--remove_h"
        VERSION_NAME="no-H"
    else
        H_FLAG="--with_h"
        VERSION_NAME="with-H"
    fi

    LOG_FILE="$LOG_DIR/${VERSION}_seed${SEED}_gpu${GPU}.log"

    echo "[$(date '+%H:%M:%S')] Starting Round $((i/2+1)) Task $((i%2+1)): $VERSION_NAME seed=$SEED on GPU $GPU" | tee -a "$LOG_DIR/master.log"

    TL_BACKEND="torch" CUDA_VISIBLE_DEVICES=$GPU conda run -n ggl python defog_sample_only.py \
        --dataset qm9 \
        $H_FLAG \
        --save_dir ./$CKPT \
        --seed $SEED \
        --num_samples 10000 \
        --num_sample_fold 3 \
        --evaluate \
        2>&1 | tee "$LOG_FILE" &

    # 每轮2个任务，在第2个任务后等待
    if [ $((i % 2)) -eq 1 ]; then
        echo "[$(date '+%H:%M:%S')] Waiting for Round $((i/2+1)) to complete..." | tee -a "$LOG_DIR/master.log"
        wait
        echo "[$(date '+%H:%M:%S')] Round $((i/2+1)) completed." | tee -a "$LOG_DIR/master.log"
    fi
done

# 如果任务数是奇数，等待最后一个
wait

echo "=== QM9 3-Seed Evaluation Completed at $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG_DIR/master.log"

#!/bin/bash

# 数组：数据集 算法 阈值
params=(
  "cora gbp_thr 4.0e-06"
  "cora gbp_thr 4.5e-06"
  "pubmed sgc_thr 6.0e-05"
  "pubmed sgc_thr 7.5e-05"
)

# 循环执行
for param in "${params[@]}"; do
    # 分割参数（加引号，防止解析错误）
    IFS=' ' read -r -a array <<< "$param"
    DATASTR="${array[0]}"
    ALGO="${array[1]}"
    THRA="${array[2]}"
    THRW="0.0e+00"

    # 🔥 核心修复：所有变量加双引号，路径绝对正确
    for SEED in 42
    do
        OUTDIR="./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}"
        mkdir -p "${OUTDIR}"  # 引号必须加
        OUTFILE="${OUTDIR}/out.txt"
        
        # 运行Python脚本（日志写入我们创建的文件夹，彻底解决路径错误）
        python -u run_mb_gamma.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                            --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> "${OUTFILE}"
        
        # 打印进程ID，等待完成
        echo $!
        wait
    done
done
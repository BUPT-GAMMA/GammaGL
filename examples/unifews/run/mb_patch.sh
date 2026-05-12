#!/bin/bash

params=(
  "cora gbp_thr 4.0e-06"
  "cora gbp_thr 4.5e-06"
  "pubmed sgc_thr 6.0e-05"
  "pubmed sgc_thr 7.5e-05"
)


for param in "${params[@]}"; do

    IFS=' ' read -r -a array <<< "$param"
    DATASTR="${array[0]}"
    ALGO="${array[1]}"
    THRA="${array[2]}"
    THRW="0.0e+00"

    for SEED in 42
    do
        OUTDIR="./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}"
        mkdir -p "${OUTDIR}" 
        OUTFILE="${OUTDIR}/out.txt"
        
        python -u run_mlp.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                            --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> "${OUTFILE}"
        
        echo $!
        wait
    done
done
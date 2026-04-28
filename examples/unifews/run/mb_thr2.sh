ALGO=sgc_thr
for DATASTR in cora
do
    for THRA in 2.0e-04
    # for THRA in 1.0e-07 2.0e-07 5.0e-07 2.0e-06 5.0e-06 2.0e-05 5.0e-05 2.0e-04 5.0e-04 1.0e-03 2.0e-03 5.0e-03 8.0e-03 1.0e-02 1.5e-02 2.0e-02 3.0e-02 4.0e-02 5.0e-02
    do
        # for THRW in 1.0e-02 5.0e-02 1.0e-01 5.0e-01 1.0e+00
        for THRW in 6.5e-01 7.0e-01
        do
            SEED=26
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} &
            echo $! && wait
        done
    done
done

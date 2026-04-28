ALGO=gcn_thr
for DATASTR in ogbl-collab
do
    for THRA in 0.0e+00 5.0e-02 1.0e-01 2.0e-01 3.0e-01 4.0e-01 5.0e-01 6.0e-01 7.0e-01 8.0e-01 9.0e-01 1.0e+00 1.1e+00 1.2e+00 1.5e+00 2.0e+00  # gcn_thr
    # for THRA in 0.0e+00 2.0e-01 4.0e-01 5.0e-01 6.0e-01 7.0e-01 8.0e-01 9.0e-01 1.0e+00 1.1e+00 1.2e+00 1.4e+00 1.6e+00 1.8e+00 2.0e+00 2.3e+00 2.6e+00 3.0e+00 # gcn_thr
    # for THRA in 0.0e+00 5.0e-03 1.0e-02 1.5e-02 2.0e-02 2.5e-02 3.0e-02 4.0e-02 5.0e-02 6.0e-02 8.0e-02 1.0e-01 1.5e-01 2.0e-01 4.0e-01 5.0e-01 7.5e-01 1.0e+00 # gat_thr
    do
        # for THRW in 0.0e+00 5.0e-02 1.0e-01 2.0e-01 3.0e-01 4.0e-01 5.0e-01 6.0e-01 7.0e-01 8.0e-01 9.0e-01 1.0e+00 1.2e+00 1.4e+00 1.7e+00 2.0e+00
        for THRW in 0.0e+00 #8.0e-01 1.2e+00 2.0e+00
        # for THRW in 6.0e-02 7.5e-02 8.5e-02 2.5e-01 3.0e-01 3.5e-01
        do
            for SEED in 41
            do
                OUTDIR=./save/${DATASTR}-lp/${ALGO}/${SEED}-${THRA}-${THRW}
                mkdir -p ${OUTDIR}
                OUTFILE=${OUTDIR}/out.txt
                python -u run_lp.py --seed ${SEED} --config ./config/${DATASTR}_lp.json --dev ${1:--1} \
                    --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
                echo $! && wait
            done
        done
    done
done

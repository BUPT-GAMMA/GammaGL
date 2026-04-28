for DATASTR in cora
do
    for ALGO in gbp_thr
    do
        # for THRA in 1.0e-07 2.0e-07 5.0e-07 2.0e-06 5.0e-06 2.0e-05 5.0e-05 2.0e-04 5.0e-04 1.0e-03 2.0e-03 5.0e-03 8.0e-03 1.0e-02 1.5e-02 2.0e-02 3.0e-02 4.0e-02 5.0e-02
        for THRA in 0.0e+00 1.0e-06 2.0e-06 1.0e-05 5.0e-05
        # for THRA in 1.0e-07 5.0e-07 7.0e-07 1.0e-06 2.0e-06 5.0e-06 7.0e-06 1.0e-05 2.0e-05 5.0e-05 2.0e-04 5.0e-04 1.0e-03 2.0e-03 5.0e-03 8.0e-03 1.0e-02 2.0e-02 5.0e-02 # gbp citeseer
        # for THRA in 0.0e+00 5.0e-07 1.0e-05 1.0e-04 1.0e-03 5.0e-03 1.0e-02 # sgc citeseer
        # for THRA in 0.0e+00 1.0e-07 5.0e-07 1.0e-06 1.0e-05 5.0e-05 1.0e-04 5.0e-04 1.0e-03 2.0e-03 5.0e-03 1.0e-02 5.0e-02 # sgc citeseer
        # for THRA in 0.0e+00 1.0e-03 2.0e-03 5.0e-03 8.0e-03 1.0e-02 1.5e-02 2.0e-02 2.5e-02 3.0e-02 3.5e-02 4.0e-02 4.5e-02 5.0e-02 # sgc cora
        # for THRA in 0.0e+00 1.0e-04 2.0e-04 5.0e-04 8.0e-04 1.0e-03 1.5e-03 2.0e-03 2.5e-03 3.0e-03 3.5e-03 4.0e-03 4.5e-03 5.0e-03 # gbp
        do
            for THRW in 0.0e+00 5.0e-02 1.5e-01 3.0e-01 5.0e-01 7.0e-01
            # for THRW in 4.0e-01 5.0e-01 6.0e-01 7.0e-01 8.0e-01 9.0e-01 1.0e+00 1.2e+00 1.5e+00 2.0e+00 # sgc citeseer
            do
                for SEED in 42
                do
                    OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
                    mkdir -p ${OUTDIR}
                    OUTFILE=${OUTDIR}/out.txt
                    python -u run_mb_gamma.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                        --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
                    echo $! && wait
                done
            done
        done
    done
done

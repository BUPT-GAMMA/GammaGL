for DATASTR in arxiv
do
    for ALGO in sgc_thr gbp_thr
    do
        # for THRA in 1.0e-07 2.0e-07 5.0e-07 1.0e-06 2.0e-06 4.0e-06 6.0e-06 8.0e-06 1.0e-05 2.0e-05 4.0e-05 6.0e-05 1.0e-04 2.0e-04 1.0e-03 1.0e-02
        for THRA in 1.0e-08 5.0e-08 2.0e-07 5.0e-07 2.0e-06 5.0e-06 1.0e-05 4.0e-05 6.0e-05 1.0e-04
        # for THRA in 0.0e+00 1.0e-06 1.0e-05 2.0e-05 3.0e-05 2.0e-05 1.0e-04 6.0e-04 # sgc pubmed
        # for THRA in 0.0e+00 1.0e-07 5.0e-07 1.0e-06 5.0e-06 1.0e-05 5.0e-05 1.0e-04 2.0e-04 3.0e-04 4.0e-04 5.0e-04 8.0e-04 2.0e-03 # sgc pubmed
        # for THRA in 0.0e+00 1.0e-03 2.0e-03 5.0e-03 8.0e-03 1.0e-02 1.5e-02 2.0e-02 2.5e-02 3.0e-02 3.5e-02 4.0e-02 4.5e-02 5.0e-02 # sgc
        # for THRA in 0.0e+00 1.0e-04 2.0e-04 5.0e-04 8.0e-04 1.0e-03 1.5e-03 2.0e-03 2.5e-03 3.0e-03 3.5e-03 4.0e-03 4.5e-03 5.0e-03 # gbp
        do
            for THRW in 0.0e+00 5.0e-02
            # for THRW in 0.0e+00 5.0e-02 1.0e-01 3.0e-01 5.0e-01 8.0e-01
            # for THRW in 0.0e+00 2.0e-02 1.0e-01 2.0e-01 3.0e-01 4.0e-01 5.0e-01 7.0e-01 1.0e+00 1.2e+00 1.5e+00 2.0e+00 # sgc pubmed
            do
                for SEED in 43
                do
                    OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
                    mkdir -p ${OUTDIR}
                    OUTFILE=${OUTDIR}/out.txt
                    python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                        --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
                    echo $! && wait
                done
            done
        done
    done
done

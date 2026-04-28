for DATASTR in arxiv
do
    for ALGO in sgc gbp
    do
        for SEED in 44 45 46
        do
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-0.0e+00-0.0e+00
            mkdir -p ${OUTDIR}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                --algo ${ALGO} --thr_a 1.0e-8 --thr_w 0.0 >> ${OUTFILE} &
            echo $! && wait
        done
    done
done

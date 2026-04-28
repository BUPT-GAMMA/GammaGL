for ALGO in gcn gat
do
    for DATASTR in cora citeseer pubmed
    do
        for SEED in 41 42 43
        do
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-0.0e+00-0.0e+00
            mkdir -p ${OUTDIR}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_fb_gamma.py --seed ${SEED} --config ./config/${DATASTR}.json --dev ${1:--1} \
                --algo ${ALGO} >> ${OUTFILE} &
            echo $! && wait
        done
    done
done

UNIFEWS:You Need Fewer Operations for Efficient Graph Neural Networks
============

- Paper link: [UNIFEWS:You Need Fewer Operations for Efficient Graph Neural Networks](https://arxiv.org/abs/2403.13268)

- Author's code repo:(https://github.com/gdmnl/Unifews). 



Dataset Statics
-------
| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

Results
-------
```bash
ALGO=gcn2_thr 
for DATASTR in cora citeseer pubmed
do
    for THRA in 0.0e+00 5.0e-02 
    do
        for THRW in 0.0e+00 5.0e-02 1.0e-01 
        do
            for SEED in 42 
            do
                OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
                mkdir -p ${OUTDIR}
                OUTFILE=${OUTDIR}/out.txt
                python -u run_single.py --seed ${SEED} --config ./config/${DATASTR}.json --dev ${1:--1} \
                    --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
                echo $! && wait
            done
        done
    done
done

```

-------
```bash
ALGO=gcn2_thr #gcn2
for DATASTR in cora citeseer pubmed
do
    for THRA in 0.0e+00 5.0e-03 1.0e-02 1.5e-02 2.0e-02 2.5e-02 3.0e-02 4.0e-02 5.0e-02 6.0e-02 8.0e-02 1.0e-01 1.5e-01 
    do

        for THRW in 0.0e+00 8.0e-01 1.2e+00 2.0e+00
       
        do
            for SEED in 42 43
            do
                OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
                mkdir -p ${OUTDIR}
                OUTFILE=${OUTDIR}/out.txt
                python -u run_fb.py --seed ${SEED} --config ./config/${DATASTR}.json --dev ${1:--1} \
                    --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
                echo $! && wait
            done
        done
    done
done


```
| dataset  |paper original|our original |paper unifews|our unifews  |
|----------|--------------|-------------|-------------|-------------|
| cora     |88.59(±0.52)  |82.74(±1.42) |87.06(±0.51) |85.13(±0.16) |
| citeseer |75.35(±0.80)  |73.229(±0.00)|71.03(±2.06) |72.15(±0.00) |
| pubmed   |85.77(±0.65)  |88.276(±0.00)|88.38(±0.29) |89.62(±0.00) |
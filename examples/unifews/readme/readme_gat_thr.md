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
ALGO=gat_unifews #gat
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



```
| dataset  |paper original|our original |paper unifews|our unifews  |
|----------|--------------|-------------|-------------|-------------|
| cora     |86.44(±0.55)  |87.73(±0.79) |86.20(±1.14) |88.59(±0.00) |
| citeseer |71.55(±1.52)  |73.19(±0.56) |69.97(±2.63) |73.71(±0.00) |
| pubmed   |84.56(±0.37)  |84.34(±0.41) |80.08(±4.65) |84.44(±0.00) |
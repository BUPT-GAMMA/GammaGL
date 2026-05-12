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
ALGO=gcn_unifews #gcn
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
| cora     |88.37(±0.95)  |82.74(±1.42) |87.06(±1.46) |87.94(±0.00) |
| citeseer |74.07(±0.91)  |72.23(±0.62) |71.37(±3.50) |74.91(±0.00) |
| pubmed   |84.75(±0.51)  |83.41(±0.51) |81.68(±6.07) |84.58(±0.00) |
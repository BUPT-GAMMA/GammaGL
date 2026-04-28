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
ALGO=gsage_thr 
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
ALGO=gsage_thr #gsage
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
| cora     |88.21(±0.15)  |88.585(±0.00)|86.03(±1.89) |88.26(±0.00) |
| citeseer |73.11(±1.41)  |74.310(±0.00)|72.41(±2.00) |76.35(±0.00) |
| pubmed   |88.22(±0.56)  |88.012(±0.00)|84.54(±2.92) |88.49(±0.03) |

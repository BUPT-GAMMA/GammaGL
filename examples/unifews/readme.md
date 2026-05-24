# UNIFEWS: You Need Fewer Operations for Efficient Graph Neural Networks

- Paper link: [UNIFEWS: You Need Fewer Operations for Efficient Graph Neural Networks](https://arxiv.org/abs/2403.13268)
- Author's code repo: [https://github.com/gdmnl/Unifews](https://github.com/gdmnl/Unifews)

## Verified Backend

Currently verified: **Torch** only. Other backends (Paddle, TensorFlow, MindSpore) are not yet tested.

## Dependencies

```bash
pip install dotmap
```

Optional for FLOPs counting:
```bash
pip install ptflops
```

## Dataset Statistics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |

## Iterate Mode (run_single.py)

Iterate mode trains GNNs with integrated pruning. **No Cython compilation required.**

```bash
ALGO=gat_unifews  # gat, gcn, gcn_unifews, gcn2, gcn2_unifews, gsage, gsage_unifews
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
                TL_BACKEND="torch" python -u run_single.py --seed ${SEED} --dev ${1:--1} --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} --data ${DATASTR} --path ./data/ --epochs 200 --patience 20 --lr 0.001 --weight_decay 1e-5 --layer 2 --hidden 512 --dropout 0.5 --inductive false --multil false >> ${OUTFILE} &
                echo $! && wait
            done
        done
    done
done
```

## Decouple Mode (run_mlp.py)

Decouple mode uses precomputed propagation matrices with Cython acceleration.
**Requires building the Cython extension first:**

```bash
pip install Cython eigency
cd examples/unifews/precompute
python setup.py build_ext --inplace
cd ..
```

Then run the experiment:

```bash
for DATASTR in cora
do
    for ALGO in gbp_thr  # sgc
    do
        for THRA in 0.0e+00 1.0e-06 2.0e-06 1.0e-05 5.0e-05
        do
            for THRW in 0.0e+00 5.0e-02 1.5e-01 3.0e-01 5.0e-01 7.0e-01
            do
                for SEED in 42
                do
                    OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
                    mkdir -p ${OUTDIR}
                    OUTFILE=${OUTDIR}/out.txt
                    TL_BACKEND="torch" python -u run_mlp.py --seed ${SEED} --dev ${1:--1} --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} --data ${DATASTR} --path ./data/ --epochs 200 --patience 20 --batch 512 --lr 0.01 --weight_decay 1e-5 --layer 2 --hidden 512 --dropout 0.5 --inductive false --multil false --hop 20 --alpha 0.1 --rrz 0.5 >> ${OUTFILE} &
                    echo $! && wait
                done
            done
        done
    done
done
```

## Data Preparation

Standard Cora/Citeseer/Pubmed datasets are downloaded automatically.

For OGB or PyG datasets, use the conversion script:
```bash
python convert_data.py --dataset ogbn-papers100M --dataset-root /path/to/dataset --output-path ./data/
```

## Results

Results below are from single-seed runs. Backend: Torch, seed: 42.

### GAT
| dataset  | paper original | our original | paper unifews | our unifews |
|----------|---------------|-------------|--------------|------------|
| cora     | 86.44 (+/-0.55) | 87.73 (+/-0.79) | 86.20 (+/-1.14) | 88.59 |
| citeseer | 71.55 (+/-1.52) | 73.19 (+/-0.56) | 69.97 (+/-2.63) | 73.71 |
| pubmed   | 84.56 (+/-0.37) | 84.34 (+/-0.41) | 80.08 (+/-4.65) | 84.44 |

### GCN
| dataset  | paper original | our original | paper unifews | our unifews |
|----------|---------------|-------------|--------------|------------|
| cora     | 88.37 (+/-0.95) | 82.74 (+/-1.42) | 87.06 (+/-1.46) | 87.94 |
| citeseer | 74.07 (+/-0.91) | 72.23 (+/-0.62) | 71.37 (+/-3.50) | 74.91 |
| pubmed   | 84.75 (+/-0.51) | 83.41 (+/-0.51) | 81.68 (+/-6.07) | 84.58 |

### GCNII
| dataset  | paper original | our original | paper unifews | our unifews |
|----------|---------------|-------------|--------------|------------|
| cora     | 88.59 (+/-0.52) | 82.74 (+/-1.42) | 87.06 (+/-0.51) | 85.13 |
| citeseer | 75.35 (+/-0.80) | 73.23         | 71.03 (+/-2.06) | 72.15 |
| pubmed   | 85.77 (+/-0.65) | 88.28         | 88.38 (+/-0.29) | 89.62 |

### GraphSAGE
| dataset  | paper original | our original | paper unifews | our unifews |
|----------|---------------|-------------|--------------|------------|
| cora     | 88.21 (+/-0.15) | 88.59         | 86.03 (+/-1.89) | 88.26 |
| citeseer | 73.11 (+/-1.41) | 74.31         | 72.41 (+/-2.00) | 76.35 |
| pubmed   | 88.22 (+/-0.56) | 88.01         | 84.54 (+/-2.92) | 88.49 |

# GraphTokenizer

This example provides graph serialization, Graph BPE tokenization, masked-language-model pretraining, supervised fine-tuning, checkpointing, and multi-seed evaluation for GraphTokenizer.

## Paper

GraphTokenizer is described in [Graph Tokenization for Bridging Graphs and Transformers](https://openreview.net/forum?id=jCctxI1BGF) (ICLR 2026).

The paper protocol uses frequency-guided Eulerian serialization (Feuler), fits Graph BPE on the training split only, pretrains with masked language modeling, selects the best fine-tuning checkpoint by validation performance, evaluates the test split once, and reports the mean and population standard deviation over five runs.

The current serializer supports the forward graph-to-token path only. `deserialize()` is intentionally not implemented, so this example must not be described as a reversible graph↔token interface. The native TLX GTE is randomly initialized and is distinct from the paper's pretrained GTE checkpoint.

## Datasets

The paper commands below cover:

- QM9: joint 16-target regression, reported with MAE.
- OGBG-molhiv: binary classification, reported with OGB ROC-AUC.
- Peptides-struct: 11-target regression, reported with Average MAE.

The GammaGL dataset classes download the official GraphTokenizer release bundle on first use. The bundle is cached under `<data-root>/.graph_tokenizer_release`, then copied into each dataset's normal `raw/` directory. An existing archive or extracted release can be reused with `GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/path/to/bundle-or-directory`.

## Requirements

Paper mode requires:

- `TL_BACKEND=torch`
- PyTorch 2.1.2 with CUDA 12.1
- DGL 2.4.0 with CUDA 12.1
- PyTorch Geometric 2.4.0
- TensorLayerX, NumPy, OGB, and the GammaGL package
- the native Graph BPE extension for the commands below

Build the native Graph BPE backend from the repository root:

```bash
export TL_BACKEND=torch
export DGLBACKEND=pytorch
python third_party/graph_bpe_cpp/setup.py build_ext --inplace
```

The paper models use public TensorLayerX layers and operations directly; Hugging Face Transformers is not required by GraphBERT or GraphGTE.

## How to Run

All paper hyperparameters are passed explicitly through `argparse`. The commands use repository-relative data, cache, and result paths and can be run directly from the GammaGL repository root.

### QM9 + BERT

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset qm9 \
    --model bert \
    --serialization feuler \
    --pretrain-epoch 200 \
    --n-epoch 200 \
    --pretrain-lr 0.0001 \
    --lr 0.00001 \
    --batch-size 32 \
    --weight-decay 0.1 \
    --pretrain-warmup-ratio 0.12 \
    --finetune-warmup-ratio 0.025 \
    --pretrain-max-grad-norm 2.0 \
    --finetune-max-grad-norm 0.5 \
    --mask-prob 0.09 \
    --patience 20 \
    --max-position-embeddings 768 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/qm9_bert
```

### QM9 + GTE

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset qm9 \
    --model gte \
    --serialization feuler \
    --pretrain-epoch 200 \
    --n-epoch 200 \
    --pretrain-lr 0.00005 \
    --lr 0.00001 \
    --batch-size 32 \
    --weight-decay 0.1 \
    --pretrain-warmup-ratio 0.12 \
    --finetune-warmup-ratio 0.025 \
    --pretrain-max-grad-norm 2.0 \
    --finetune-max-grad-norm 0.5 \
    --mask-prob 0.09 \
    --patience 20 \
    --max-position-embeddings 8192 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/qm9_gte
```

### OGBG-molhiv + BERT

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset OGBG-molhiv \
    --model bert \
    --serialization feuler \
    --pretrain-epoch 200 \
    --n-epoch 200 \
    --pretrain-lr 0.0001 \
    --lr 0.00005 \
    --batch-size 32 \
    --weight-decay 0.1 \
    --pretrain-warmup-ratio 0.12 \
    --finetune-warmup-ratio 0.025 \
    --pretrain-max-grad-norm 2.0 \
    --finetune-max-grad-norm 0.5 \
    --mask-prob 0.09 \
    --patience 20 \
    --max-position-embeddings 768 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/molhiv_bert
```

### OGBG-molhiv + GTE

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset OGBG-molhiv \
    --model gte \
    --serialization feuler \
    --pretrain-epoch 200 \
    --n-epoch 200 \
    --pretrain-lr 0.00005 \
    --lr 0.00005 \
    --batch-size 32 \
    --weight-decay 0.1 \
    --pretrain-warmup-ratio 0.12 \
    --finetune-warmup-ratio 0.025 \
    --pretrain-max-grad-norm 2.0 \
    --finetune-max-grad-norm 0.5 \
    --mask-prob 0.09 \
    --patience 20 \
    --max-position-embeddings 8192 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/molhiv_gte
```

### Peptides-struct + BERT

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset Peptides-struct \
    --model bert \
    --serialization feuler \
    --pretrain-epoch 200 \
    --n-epoch 200 \
    --pretrain-lr 0.0001 \
    --lr 0.00001 \
    --batch-size 16 \
    --weight-decay 0.1 \
    --pretrain-warmup-ratio 0.12 \
    --finetune-warmup-ratio 0.025 \
    --pretrain-max-grad-norm 2.0 \
    --finetune-max-grad-norm 0.5 \
    --mask-prob 0.09 \
    --patience 20 \
    --max-position-embeddings 768 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/peptides_struct_bert
```

### Peptides-struct + GTE

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset Peptides-struct \
    --model gte \
    --serialization feuler \
    --pretrain-epoch 200 \
    --n-epoch 200 \
    --pretrain-lr 0.0001 \
    --lr 0.00001 \
    --batch-size 16 \
    --weight-decay 0.1 \
    --pretrain-warmup-ratio 0.12 \
    --finetune-warmup-ratio 0.025 \
    --pretrain-max-grad-norm 2.0 \
    --finetune-max-grad-norm 0.5 \
    --mask-prob 0.09 \
    --patience 20 \
    --max-position-embeddings 8192 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/peptides_struct_gte
```

Before a long run, add `--preflight` to the corresponding command to validate the dataset, model, runtime, and BPE backend without training. `--resume` restores the latest phase, optimizer, scheduler, random-number-generator, and AMP scaler state from `last_state.pt`; `best.pt` remains the lightweight best-model checkpoint.

For a small non-paper smoke test:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --smoke \
    --dataset qm9 \
    --model bert \
    --data-root data \
    --bpe-backend python
```

## Results

The paper reports the following five-run mean results:

| Dataset | Encoder | Metric | Paper | GammaGL status |
| --- | --- | --- | ---: | --- |
| QM9 | BERT | raw MAE ↓ | 0.122 | revalidation required |
| QM9 | GTE | raw MAE ↓ | 0.071 | revalidation required: official weights |
| OGBG-molhiv | BERT | ROC-AUC ↑ | 82.6% | revalidation required |
| OGBG-molhiv | GTE | ROC-AUC ↑ | 87.4% | revalidation required: official weights |
| Peptides-struct | BERT | Average MAE ↓ | 0.247 | revalidation required |
| Peptides-struct | GTE | Average MAE ↓ | 0.242 | revalidation required: official weights |

`Paper` is the value reported by the GraphTokenizer paper. GammaGL results must not be compared or described as reproduced until the strict run records the official GTE checkpoint provenance and reports QM9 in raw label units.

Each run keeps result artifacts under `--output-dir`, including the summary JSON, per-run CSV, Markdown/LaTeX paper tables, runtime manifest, checkpoints, and paper-protocol state. JSON remains an output format only; it is not used as an input parameter configuration.

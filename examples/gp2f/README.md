# GP2F

This directory reproduces the Cora-to-PubMed 1-shot result of **GP2F** using
GammaGL with the PyTorch backend. The implementation keeps the released
objective and evaluates its all-pairs losses by row chunks, avoiding PubMed's
otherwise prohibitive dense 19717 x 19717 similarity matrices.

- Paper: https://arxiv.org/abs/2602.11629
- Official implementation: https://github.com/hedongxiao-tju/GP2F

## Requirements

Install GammaGL with the GAMMA Lab TensorLayerX nightly build and run the
commands from the repository root. This reproduction currently requires the
PyTorch backend.

## Directory contents

- `grace_pretrain_trainer.py`: pretrains one frozen Cora GRACE encoder.
- `gp2f_trainer.py`: tunes GP2F on PubMed and optionally saves per-trial
  test accuracies through `--result_path`.

## Cora GRACE pretraining

The official protocol pretrains an independent Cora encoder for every outer
seed. The checkpoint contains only the frozen encoder, as in the official
code.

    TL_BACKEND=torch python examples/gp2f/grace_pretrain_trainer.py \
      --gpu 0 --seed 12345 --n_epoch 1000 --patience 100 \
      --save_path output/gp2f/grace_cora_seed12345.npz

## PubMed 1-shot tuning

Run 100 PubMed trials with the checkpoint trained for the same outer seed.
sim_chunk_size changes only the evaluation order of exact loss rows. Use the
largest value that fits the available GPU memory; 4096 was suitable for the
8-GB GPU used in this reproduction.

    TL_BACKEND=torch python examples/gp2f/gp2f_trainer.py \
      --gpu 0 --pretrained_path output/gp2f/grace_cora_seed12345.npz \
      --seeds 12345 --runs 100 --n_epoch 500 --sim_chunk_size 4096 \
      --result_path output/gp2f/results/pubmed_1shot_seed12345.npy

Repeat the two commands for 23456, 34567, 45678, and 56789. Concatenate
the five saved arrays to report the final 500-run statistic:

    python - <<'PY'
    import glob
    import numpy as np

    paths = sorted(glob.glob('output/gp2f/results/pubmed_1shot_seed*.npy'))
    scores = np.concatenate([np.load(path) for path in paths])
    print('GP2F Cora->PubMed 1-shot: {:.2f} +/- {:.2f} ({} runs)'.format(
        scores.mean(), scores.std(), len(scores)))
    PY

## Result

| Source -> target | Setting | Paper | GammaGL (Torch) |
|---|---|---:|---:|
| Cora -> PubMed | 1-shot, 500 runs | 53.05 +/- 10.64 | 52.69 +/- 10.96 |

The GammaGL result is calculated from five independently pretrained Cora
encoders and 100 PubMed trials for each encoder.

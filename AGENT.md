# GammaGL Agent Guide

This repository contains GammaGL, a TensorLayerX-based graph learning library.
When a coding CLI works in this repository, use the local GammaGL developer skill:

```text
skills/gammagl-developer/SKILL.md
```

The skill explains the repository layout, installation paths, common examples,
public API update rules, and troubleshooting notes for TensorLayerX, CUDA ops,
optional dependencies, and datasets.

## Environment Rules

- GammaGL uses the GAMMA Lab TensorLayerX branch, not the upstream PyPI package:

  ```bash
  pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
  ```

- Install a backend before installing TensorLayerX. For the default PyTorch
  backend, install CPU or CUDA PyTorch first, then install the TensorLayerX
  nightly branch.
- Do not force `TL_BACKEND` in library code at import time. Users and tests
  choose the backend through the environment.
- Keep heavy LLM/GFM dependencies optional. Core imports must not require
  `transformers`, `torch_geometric`, `openai`, or `sentence_transformers`.

## Install Paths

CPU baseline:

```bash
conda create -n gammagl-cpu python=3.10
conda activate gammagl-cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=0 pip install -e ".[build]" --no-build-isolation
```

CUDA baseline:

```bash
conda create -n gammagl-cu python=3.10
conda activate gammagl-cu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=auto pip install -e ".[build]" --no-build-isolation
```

LLM/GFM extension:

```bash
pip install pybind11 ninja
pip install -e ".[llm-gfm]" --no-build-isolation
```

## Routine Checks

Use the installed environment for checks:

```bash
TL_BACKEND=torch python -m compileall -q gammagl tests examples
TL_BACKEND=torch python -c "import gammagl; print(gammagl.__version__)"
TL_BACKEND=torch python -c "from gammagl.layers import MessagePassing; from gammagl.models import *"
```

For examples, prefer a tiny smoke run or `--help` before running full training.
Avoid downloads and long jobs unless the task explicitly requires end-to-end
training validation.

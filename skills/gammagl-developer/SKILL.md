# GammaGL Developer Skill

Use this skill when editing GammaGL source, examples, tests, installation
metadata, or documentation.

## Repository Layout

- `gammagl/`: library package.
- `gammagl/layers/`: reusable layers. Convolution layers live in
  `gammagl/layers/conv/`.
- `gammagl/models/`: model definitions exported through
  `gammagl.models.__all__`.
- `gammagl/datasets/`: dataset wrappers exported through
  `gammagl.datasets.__all__`.
- `gammagl/utils/`: public utility functions exported through
  `gammagl.utils.__all__`.
- `examples/`: runnable model examples and reproduction scripts.
- `tests/`: unit and smoke tests.
- `setup.py`, `requirements.txt`, `docs/requirements.txt`: package and
  environment dependency entry points.

## TensorLayerX Constraint

GammaGL relies on the GAMMA Lab TensorLayerX branch:

```bash
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
```

Install the backend first. For the default torch backend, install PyTorch before
installing TensorLayerX. Do not replace this branch with the public PyPI
TensorLayerX package when validating changes.

## Quick Start Environments

CPU:

```bash
conda create -n gammagl-cpu python=3.10
conda activate gammagl-cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=0 pip install -e ".[build]" --no-build-isolation
```

CUDA:

```bash
conda create -n gammagl-cu python=3.10
conda activate gammagl-cu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=auto pip install -e ".[build]" --no-build-isolation
```

Use one GammaGL package for CPU and GPU. Select CUDA extension compilation with
`GAMMAGL_WITH_CUDA=0`, `1`, or `auto`.

LLM/GFM features are optional:

```bash
pip install pybind11 ninja
pip install -e ".[llm-gfm]" --no-build-isolation
```

These extras may install packages such as PyTorch, Transformers,
Torch-Geometric, OpenAI SDK, and Sentence Transformers. Core GammaGL imports
must not require them.

## Running Examples

Prefer `TL_BACKEND=torch` unless the task targets another backend.

```bash
TL_BACKEND=torch python examples/gcn/gcn_trainer.py --dataset cora --n_epoch 1
TL_BACKEND=torch python examples/gat/gat_trainer.py --dataset cora --n_epoch 1
TL_BACKEND=torch python examples/graphsage/train_full_trainer.py --dataset cora --n_epoch 1
```

If a trainer does not support a short epoch argument, first run `--help` and
add a small smoke path instead of forcing a full experiment.

## Public API Update Rules

When adding a layer:

- Add the implementation under the appropriate `gammagl/layers/` subpackage.
- Export public classes from the subpackage `__init__.py`.
- If docs use `from gammagl.layers import Name`, export it in
  `gammagl/layers/__init__.py`.
- Add or update a focused layer test.

When adding a model:

- Add the implementation under `gammagl/models/`.
- Export the public class from `gammagl/models/__init__.py`.
- Keep `__all__` spelling and capitalization exactly aligned with imported
  names.
- Add a focused model test or a tiny example smoke test.

When adding a dataset:

- Add the wrapper under `gammagl/datasets/`.
- Export it from `gammagl/datasets/__init__.py`.
- Avoid downloads during import. Download only when the dataset is constructed
  or processed.

When adding a utility:

- Export public utilities from `gammagl/utils/__init__.py`.
- Do not import optional heavy frameworks at module import time.
- Add a unit test for shared behavior.

## Dependency Rules

- Keep runtime dependencies small and backend-neutral.
- Keep `pytest`, `ruff`, `pybind11`, and `ninja` in development/build extras,
  not the base runtime set.
- Keep molecular dependencies such as `rdkit` optional unless a core API truly
  requires them.
- Keep LLM/GFM dependencies in extras. Do not import them from package
  `__init__` files.
- Do not add personal paths, hard-coded dataset roots, or import-time
  `TL_BACKEND` overrides.

## Troubleshooting

TensorLayerX import fails:

- Confirm PyTorch or the selected backend is installed first.
- Confirm TensorLayerX came from `dddg617/tensorlayerx.git@nightly`.

CUDA extension build fails:

- Use `GAMMAGL_WITH_CUDA=0` for CPU-only installs.
- Use `GAMMAGL_WITH_CUDA=auto` when CUDA may or may not be present.
- Use `GAMMAGL_WITH_CUDA=1` only when CUDA headers and `nvcc` are available.

Optional dependency is missing:

- Keep the error local to the optional feature.
- Add a clear installation hint for the relevant extra or package.
- Do not move the dependency into the core import path.

Dataset download fails:

- Keep tests on synthetic or cached tiny data where possible.
- Document dataset paths and credentials clearly in the example README.

## Standard Checks

```bash
TL_BACKEND=torch python -m compileall -q gammagl tests examples
TL_BACKEND=torch python -c "from gammagl.layers import MessagePassing; from gammagl.models import *"
TL_BACKEND=torch python -m pytest tests/test_public_api.py -q
```

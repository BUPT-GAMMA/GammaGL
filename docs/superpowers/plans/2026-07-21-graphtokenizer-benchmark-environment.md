# GraphTokenizer Benchmark Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated CUDA-enabled Python environment under `/local/wrq` that can preflight and run GammaGL's GraphTokenizer paper protocol for QM9, OGBG-molhiv, and Peptides-struct.

**Architecture:** Use a prefix-based Conda environment with Python 3.10 and a CUDA 12.8 PyTorch wheel; keep the GammaGL checkout uninstalled and invoke it from `/home/wrq/GammaGL` so its active, user-owned worktree is exercised directly. Add only paper-protocol dependencies, build the local C++ GraphBPE extension in place, then validate CUDA, imports, datasets and metric aggregation using the existing trainer commands.

**Tech Stack:** Conda, Python 3.10, PyTorch 2.11.0 CUDA 12.8, TensorLayerX nightly, Transformers, OGB, NumPy, Pybind11, GCC/C++17.

## Global Constraints

- Environment prefix: `/local/wrq/gammagl-graphtokenizer`; never modify or delete an existing prefix.
- All validation commands run from `/home/wrq/GammaGL` with `TL_BACKEND=torch`.
- Paper protocol uses CUDA and five independent seeds 42 through 46.
- Required benchmark metrics are QM9 MAE, OGBG-molhiv ROC-AUC, and Peptides-struct Average MAE, each reported as mean and population standard deviation across runs.
- Do not install TensorFlow, PaddlePaddle, or the repository's all-backend `requirements.txt`; they are not required by `--protocol paper` and would introduce incompatible dependency constraints.
- Preserve all pre-existing checkout changes; this task writes no GammaGL source code and creates no Git commit.

---

### Task 1: Create and Populate the Isolated Runtime

**Files:**
- Create: Conda prefix `/local/wrq/gammagl-graphtokenizer`

**Interfaces:**
- Produces: `/local/wrq/gammagl-graphtokenizer/bin/python`, a Python 3.10 interpreter that imports CUDA PyTorch, TensorLayerX, Transformers, OGB and Pybind11.

- [x] **Step 1: Confirm the target prefix does not exist and inspect the available NVIDIA driver.**

Run: `test ! -e /local/wrq/gammagl-graphtokenizer && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`

Expected: target absent and RTX 3090 GPUs visible to driver 580.159.03.

- [x] **Step 2: Create the Python 3.10 Conda prefix.**

Run: `/home/wrq/miniconda3/bin/conda create --yes --prefix /local/wrq/gammagl-graphtokenizer python=3.10 pip setuptools wheel`

Expected: Conda reports a completed transaction and the prefix contains `bin/python`.

- [x] **Step 3: Install the CUDA runtime and paper-only dependencies.**

Run: `/local/wrq/gammagl-graphtokenizer/bin/python -m pip install --upgrade pip && /local/wrq/gammagl-graphtokenizer/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && /local/wrq/gammagl-graphtokenizer/bin/python -m pip install 'numpy==1.23.5' 'git+https://github.com/dddg617/TensorLayerX.git@nightly' 'transformers>=4.40,<5' 'ogb>=1.3.6' 'pybind11>=2.11' tqdm rich packaging psutil`

Expected: all package installs finish successfully; PyTorch records a CUDA 12.8 runtime.

- [x] **Step 4: Verify core imports and CUDA availability.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python -c "import torch, tensorlayerx, transformers, ogb, pybind11; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"`

Expected: exit code zero and a CUDA-enabled PyTorch version plus `NVIDIA GeForce RTX 3090`.

### Task 2: Build the Local Extension and Validate the Protocol

**Files:**
- Create: `/home/wrq/GammaGL/third_party/graph_bpe_cpp/_graph_bpe*.so` (untracked build artifact)

**Interfaces:**
- Consumes: the runtime from Task 1 and active GammaGL worktree.
- Produces: an importable `third_party.graph_bpe_cpp._graph_bpe` extension and successful paper-mode preflight reports.

- [x] **Step 1: Build the native GraphBPE extension with the new interpreter.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python third_party/graph_bpe_cpp/setup.py build_ext --inplace`

Expected: compilation completes and the extension is written beneath `third_party/graph_bpe_cpp`.

- [x] **Step 2: Verify the GammaGL imports and native BPE backend.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python -c "from gammagl import datasets; from third_party import graph_bpe_cpp; assert graph_bpe_cpp.is_available(); print('GraphBPE ready')"`

Expected: `GraphBPE ready` with no TensorLayerX backend error.

- [x] **Step 3: Run paper preflight for every required dataset.**

Run: `for dataset_args in '--dataset qm9 --target-property homo' '--dataset OGBG-molhiv' '--dataset Peptides-struct'; do TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python examples/graph_tokenizer/graph_tokenizer_trainer.py --protocol paper --preflight $dataset_args --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp; done`

Expected: each command validates its official cached/downloaded source, split files, CUDA runtime and native BPE backend without beginning training.

- [x] **Step 4: Run one synthetic smoke test to verify metric formatting.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python examples/graph_tokenizer/graph_tokenizer_trainer.py --smoke --dataset p-struct --model bert`

Expected: exit code zero and a summary that labels the primary metric `average_mae`.

### Task 3: Record the Full Benchmark Invocation

**Files:**
- Create: `/local/wrq/graph-tokenizer-benchmark/` (runtime outputs when the user launches the benchmark)

**Interfaces:**
- Consumes: preflight-validated environment, data cache and extension.
- Produces: one summary JSON/CSV/table/manifest per dataset, with the five-run mean and population standard deviation of the required primary metric.

- [ ] **Step 1: Use the QM9 paper command, selecting one of its 16 target properties.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python examples/graph_tokenizer/graph_tokenizer_trainer.py --protocol paper --dataset qm9 --target-property homo --paper-config examples/graph_tokenizer/configs/qm9_paper.json --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp --runs 5 --output-dir /local/wrq/graph-tokenizer-benchmark/qm9_homo`

Expected: `qm9_bert_summary.json` records primary metric `mae`, plus `test_mae_mean` and `test_mae_std` over seeds 42–46.

- [ ] **Step 2: Run the OGBG-molhiv paper command.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python examples/graph_tokenizer/graph_tokenizer_trainer.py --protocol paper --dataset OGBG-molhiv --paper-config examples/graph_tokenizer/configs/molhiv_paper.json --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp --runs 5 --output-dir /local/wrq/graph-tokenizer-benchmark/molhiv`

Expected: `molhiv_bert_summary.json` records primary metric `rocauc`, plus `test_rocauc_mean` and `test_rocauc_std` over seeds 42–46.

- [ ] **Step 3: Run the Peptides-struct paper command.**

Run: `TL_BACKEND=torch /local/wrq/gammagl-graphtokenizer/bin/python examples/graph_tokenizer/graph_tokenizer_trainer.py --protocol paper --dataset Peptides-struct --paper-config examples/graph_tokenizer/configs/peptides_struct_paper.json --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp --runs 5 --output-dir /local/wrq/graph-tokenizer-benchmark/peptides_struct`

Expected: `peptides_struct_bert_summary.json` records primary metric `average_mae`, plus `test_average_mae_mean` and `test_average_mae_std` over seeds 42–46.

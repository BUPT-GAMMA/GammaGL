# GraphTokenizer 训练

`graph_tokenizer_trainer.py` 是训练入口。它使用 GammaGL 的 Feuler 序列化器、Graph BPE、
tokenizer、数据集以及原生 TensorLayerX `GraphBERT` / `GraphGTE`。

正式路径默认以种子 `42 43 44 45 46` 完成五次彼此独立的完整训练：仅在训练集拟合
BPE／标签归一化，在完整训练集上进行 MLM，再微调、验证、恢复最佳微调 checkpoint 并最终
测试。最终指标为五次最终测试的 `mean ± std`（样本标准差，`ddof=1`）。测试集只会在恢复
最佳 checkpoint 后评估。

长度超过 `max_length` 的序列化图会被拒绝而非截断，因为截断会丢失图结构。CLI 参数会覆盖
`PAPER_CONFIGS` 中对应 preset 的值。

## 正式 benchmark 与结果

以下六组 GammaGL reproduction 按用户确认，保留为此前报告的最新版五 seed 最终测试
成绩。本次安装验证和梯度累积修复没有重新运行这六组正式训练。报告 `mean ± std`，
跨 seed 标准差使用样本标准差（`ddof=1`）。论文／官方参考成绩独立列出，不作为 GammaGL 的实验记录。

| 数据集 | Encoder | 指标 | 论文／官方参考 | 当前 GammaGL reproduction |
| :-- | :-- | :-- | :-- | :-- |
| Mutagenicity | BERT | Accuracy (%) ↑ | 87.5 ± 0.9 | 76.73 ± 1.75 |
| Mutagenicity | GTE | Accuracy (%) ↑ | 90.1 ± 0.7 | 68.11 ± 5.88 |
| DBLP_v1 | BERT | Accuracy (%) ↑ | 93.2 ± 0.1 | 92.25 ± 1.28 |
| DBLP_v1 | GTE | Accuracy (%) ↑ | 93.6 ± 0.1 | 92.14 ± 0.45 |
| AQSOL | BERT | 原尺度 MAE ↓ | 0.648 ± 0.008 | 0.7395 ± 0.0072 |
| AQSOL | GTE | 原尺度 MAE ↓ | 0.609 ± 0.016 | 0.7077 ± 0.0069 |

Accuracy 的均值用百分数表示，`±` 后的标准差单位为**百分点**；相对于 trainer 输出的
0–1 Accuracy，均值和标准差均乘以 100，显示两位小数。AQSOL 的 MAE 保持原标签尺度，
显示四位小数，不做百分数转换。

### 实验记录与可追溯性

本次没有建立这六组成绩与精确运行 commit、完整生效配置、五个 seed 原始指标、日志及
checkpoint 的对应关系，不把本次 smoke 或单元测试作为这些成绩的运行凭据，也不据此
将成绩改标为旧协议结果。后续补充版本追溯时应提供可核实的原始记录。

Trainer 接受 `mutag` 和 `muta` 作为 **Mutagenicity** 的 aliases，不指 188-graph `MUTAG`。
`dblp` 对应 **DBLP_v1 graph-classification** benchmark，数据集类为 `GraphTokenizerDBLP`，
与异构节点分类数据集 `DBLP` 区分。

## 支持的图范围

`FrequencyGuidedEulerianSerializer` 面向简单无向图，覆盖这三个 GraphTokenizer
benchmark 的图结构。它支持非连通图、孤立节点及标量节点/边标签；空节点图会保留为零节点
图。显式声明有向语义的图对象会被拒绝；单向 COO 仅可表示一条无向边的存储形式。原生
GammaGL `Graph.is_directed()` 根据 COO 对称性推断存储方向，不等同于显式有向声明；
无方向声明的单向或对称 COO 均可使用，显式 `directed=True` 或 `is_directed=True` 则会被拒绝。

有向图语义、自环和并行边／多重图均不支持，会 fail fast 而不会静默转换。

### 节点引用与置换保证

Feuler 的节点引用 token 使用最终组件顺序及遍历中的首次访问顺序，从 0 开始连续编号。
重复访问同一节点复用引用，不同组件共享一套局部编号。token stream 单独可恢复同构的带标签图；
完整 serializer/tokenizer result 中的 `local_to_original` metadata 可恢复原始节点编号。
原始编号只在该 provenance metadata 中，训练 `input_ids` 不含原始编号。
缺少节点特征（`x=None`）的图使用统一的零节点标签；反序列化后的 `x` 也是零标签，
不表示恢复了原本不存在的特征。
若 held-out 图出现未见标签，模型 `input_ids` 可能含 `UNK`，不能单独逆转；完整 result
保留未损失的 `serialized_token_ids`，可结合 metadata 恢复。

序列化器不提供任意图的全局同构规范化。完整 token 与
`input_ids` 的置换一致性已对三节点示例、带标签路径和环、非连通图及孤立节点验证；显式
`start_node` 比较使用重编号后的对应节点。结构签名或组件排序键出现平局时仍可能依赖原始编号：
例如五节点同标签图的边为 `(0,1),(0,2),(0,4),(1,2),(1,3)`，按
`0→1,1→2,2→3,3→0,4→4` 重编号后，完整输入不同，多起点选择也不能逐项对应。
四环与三角形中的节点还可具有相同的结构签名，却属于不同拓扑的组件；不能仅凭签名相同
视为可互换。等长度的路径组件与三角形组件也可在组件排序键上打平。

### 输入协议与产物兼容性

协议审计基线为 `18f6ebf6c94d9738ea0b51b1ccd42d16ae4e2722`；当前 serializer protocol 为 **4**、
tokenizer schema 为 **4**、预处理缓存格式为 **5**。这个代码基线不等同于
六组实验已经核实的运行 commit。

旧 protocol 2 把原始节点编号写入引用 token；`21d93d7` 引入 protocol／schema 3，改为
遍历局部引用并修复 AQSOL 按 serialization 数量加权的归一化。`18f6ebf` 再将缺少节点特征
的默认标签从原始编号改为全零，并修复缓存对实际 processed 数据来源的跟踪。
显式整数节点标签仍按输入保留；benchmark Dataset 要求提供节点特征，`x=None` 的默认行为
是通用 serializer 的接口约定。

| 产物 | 当前代码实际执行的检查与处理 |
| :-- | :-- |
| 已拟合 tokenizer | 编码／解码前检查 `schema_version == 4`；旧版本或缺失字段会抛出 `RuntimeError`，需重新 fit。`original_token_ids` 缺失也会被拒绝。 |
| 完整 serializer／tokenizer result | 反序列化检查 metadata 中的 protocol／schema，不匹配会抛出 `ValueError`。裸 token 列表没有版本 metadata，无法据此自动识别旧协议，应从原图重新生成。 |
| 预处理缓存 | 键包含数据／split 内容摘要、输入协议和预处理配置；加载时再检查缓存格式 5 与 tokenizer schema 4。旧键不命中，旧格式 payload 不复用，重新预处理；通过临时文件和 `os.replace` 原子写入。 |
| 下游微调 checkpoint | `restore_checkpoint()` 在载入模型权重前检查 `tokenizer_schema_version == 4`，缺失或不一致会抛出 `ValueError`。相同 embedding 形状不能替代协议与词表对应关系。 |

不存在旧词表、缓存或下游 checkpoint 的自动迁移。`GraphBPE.load_codebook()` 只加载 BPE
规则，不校验 tokenizer 协议；下游 checkpoint 的 schema 检查也不验证具体词表或数据摘要。
同一 schema 的产物仍须与其拟合 tokenizer、配置和数据配套使用。正式 GTE 来源 checkpoint
是独立的预训练编码器初始化，其 revision／SHA-256／架构／tensor shape／覆盖率校验见下文。

## 固定的六组实验 preset

`PAPER_CONFIGS` 基于固定的
[GraphTokenizer 官方源码 revision
`98343a6b025a48fbb6859cd812a12b81ec3ac3cc`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)
整理为六组 benchmark presets。训练使用统一的 serializer/tokenization pipeline，默认使用
Python BPE，`cpp` 是可选 backend。
下表描述当前代码默认配置，不能替代每次运行的完整生效配置及配套记录。

所有 preset 均使用 `max_length=8096`、BPE `2000` merges / 最小频率 `2`、100 次序列化、
MLM mask 概率 `0.09`、warm-up `0.12 / 0.025`、梯度裁剪 `2.0 / 0.5`、默认 seeds `42–46`，
并使用 `gradient_accumulation_steps=1`。Graph BPE 仅在训练集拟合，故词表大小为运行时派生值。

| 实验 | 预训练 / 微调 batch | MLM / 微调 LR | MLM / 微调 epoch | 预训练 / 微调 weight decay | 耐心值 | 位置容量 | 架构 / checkpoint |
| :-- | --: | :-- | :-- | :-- | --: | --: | :-- |
| Mutagenicity + BERT | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8096 | 512 隐藏维、4 层、4 头、2048 FFN |
| Mutagenicity + GTE | 32 / 64 | `1e-4 / 5e-5` | 200 / 100 | `0.1 / 0.01` | 10 | 8192 | 768 隐藏维、12 层、12 头、3072 FFN、RoPE |
| DBLP_v1 + BERT | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8096 | 512 隐藏维、4 层、4 头、2048 FFN |
| DBLP_v1 + GTE | 32 / 64 | `1e-4 / 5e-5` | 200 / 100 | `0.1 / 0.01` | 10 | 8192 | 768 隐藏维、12 层、12 头、3072 FFN、RoPE |
| AQSOL + BERT | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8096 | 512 隐藏维、4 层、4 头、2048 FFN |
| AQSOL + GTE | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8192 | 768 隐藏维、12 层、12 头、3072 FFN、RoPE |

`--gradient-accumulation-steps` 仍是通用 CLI 功能；完整窗口与不足步数的尾窗口都逐批对
loss sum 反向传播，在窗口结束时按实际目标总数归一化梯度，再做梯度裁剪和更新。
分类按样本数、回归 MSE 按参与 loss 的元素数、MLM 按 `labels != -100` 的 token 数
归一化；训练 loss 日志也使用相同计数口径。MLM 中没有有效目标的 microbatch 不稀释
其他批次的梯度；整个窗口没有有效目标时清空梯度，跳过 optimizer 与 scheduler 更新。
两者实际 step 数保持一致，全空 epoch 的训练 loss 记为 0。

### 序列长度与位置容量

`max_length` 是包含 special token 的最终 GraphTokenizer 输入上限；模型另有
`max_position_embeddings` 容量。BERT 的两个值均为 8096，配置见固定 revision 的
[`config/default_config.yml`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/config/default_config.yml#L77-L83)。
collate 会 pad 到当前 batch 中最长的序列。

GTE 的最终输入上限为 8096、位置容量为 8192；后者来自固定源码的
[`gte_model/config.json`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/gte_model/config.json#L32)。

### GTE checkpoint 与 BPE

正式 `--encoder gte` 运行构建原生 TLX GraphGTE，并通过 GammaGL HF-to-TLX converter 加载
`Alibaba-NLP/gte-multilingual-base` revision
`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`。它验证 checkpoint SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`、官方架构、135/135
parameter coverage 和 tensor shape；任一校验失败都会停止运行。`--allow-random-gte-init`
仅用于开发，会标记 `reproduction: false`，不是论文复现实验命令。

`--bpe-backend python` 是默认值；`auto` 在可用时使用可选的 `graph_bpe_cpp` 原生扩展，否则
回退 Python；`cpp` 则要求该扩展并在缺失时 fail fast。无论 backend，BPE 只在训练 split 拟合，
验证与测试从不参与词表或回归标签统计。

## 安装

本安装流程固定使用 **Python 3.10、PyTorch 2.1.2、torchvision 0.16.2、
torchaudio 2.1.2、DGL 2.1.0、torchdata 0.7.1**。不要用主 README 中未固定版本的
通用安装命令替代本节；DGL 2.1 的 GraphBolt 动态库与 PyTorch 版本绑定。

### CPU：从新环境安装

需要 Git、支持 C++17／OpenMP 的 GCC/G++（例如 Ubuntu 的 `build-essential`），以及
Python 3.10 的开发头文件和 `venv`。先安装这些系统工具；下列命令从仓库外开始执行。
也可以先用 `conda create -n graph-tokenizer-paper python=3.10 pip` 和
`conda activate graph-tokenizer-paper` 创建环境，替代前两行 venv 命令。

```bash
python3.10 -m venv .venv-graph-tokenizer
source .venv-graph-tokenizer/bin/activate
python -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 10)'
export TL_BACKEND=torch
export DGLBACKEND=pytorch

# torch 2.1.2 的构建工具使用 pkg_resources.packaging；保留 setuptools 69.5.1。
# NumPy 与仓库要求一致，避免升级至 2.x。
python -m pip install --upgrade pip
python -m pip install setuptools==69.5.1 wheel "numpy>=1.24,<2" "pybind11>=2.10" "ninja>=1.11"
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install "numpy>=1.24,<2" \
  "tensorlayerx @ git+https://github.com/dddg617/tensorlayerx.git@nightly"

# 验证本 PR 时检出其现有源分支；必须取得 parallel_hashmap 子模块。
git clone --recursive --branch pr/graphtokenizer-paper-8-29-2 https://github.com/baihchou8787/GammaGL.git
cd GammaGL
git submodule update --init --recursive
GAMMAGL_WITH_CUDA=0 python -m pip install -e ".[graph-tokenizer-paper]" --no-build-isolation
python -m pip install pytest
```

`setup.py` 在解析 extra 的依赖前就会导入 GAMMA Lab TLX 的
`PyCppExtension`、`PyCUDAExtension` 和 `PyBuildExtension`，所以 **TLX 和 pybind11 等
构建依赖必须提前安装**，仅写 `pip install -e ".[graph-tokenizer-paper]"` 不够。
`--no-build-isolation` 让构建使用当前环境中的 backend／TLX／构建依赖。
TLX 会导入 torchvision；这里同时安装与仓库 CI 一致的匹配 torchaudio。
extra 固定 DGL／torchdata，并包含 GraphBolt 所需的 `pydantic`、`PyYAML` 以及正式
GTE checkpoint 获取／转换所需的 `huggingface-hub`、`safetensors`。
`requirements.txt` 是核心运行依赖列表，不能替代以上 backend、构建前置步骤及 paper extra。

若默认 PyPI 下载超时，可给 PyPI 安装命令追加
`--index-url https://mirrors.aliyun.com/pypi/simple`；PyTorch CPU 三件套仍使用上面的官方
CPU index。也可预下载同版本 wheels，但应核对官方索引的 SHA-256；本次验证的
PyTorch 三件套和 llvmlite 预下载文件均做了该校验。GitHub 的 TLX 源码获取独立于 PyPI 镜像。

### CUDA 使用

上面的安装和下方验证命令以 CPU 为准。CUDA 环境应在安装 TLX／GammaGL **之前**，按
[PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v212)
选择匹配驱动的 **2.1.2 / 0.16.2 / 2.1.2** CUDA wheels，并从
[DGL 2.1 官方安装说明](https://www.dgl.ai/dgl_docs/en/2.1.x/install/index.html)选择
同时匹配 PyTorch 2.1 与该 CUDA 版本的 DGL 2.1.0 构建。
这里不提供未经本次验证的 CUDA wheel 地址。

编译 GammaGL CUDA 扩展还需要匹配的本地 CUDA toolkit、`nvcc` 和 headers；仅安装
CUDA runtime wheel 不足以编译。工具链齐备后，在同一环境和仓库根目录执行：

```bash
export TL_BACKEND=torch
export DGLBACKEND=pytorch
GAMMAGL_WITH_CUDA=auto python -m pip install -e ".[graph-tokenizer-paper]" --no-build-isolation
python -c "import torch, torchvision, tensorlayerx, dgl, dgl.graphbolt, torchdata"
```

`GAMMAGL_WITH_CUDA=0` 明确构建 CPU 扩展；`auto` 根据本地 toolkit 决定；`1` 强制
构建 CUDA 扩展。CUDA 构建和正式 GPU 训练不属于下方 CPU smoke 的验证结论。

### CPU 安装验证

在已激活的 Python 3.10 环境和仓库根目录执行：

```bash
export TL_BACKEND=torch
export DGLBACKEND=pytorch
python -c "import torch, torchvision, torchaudio, tensorlayerx, dgl, dgl.graphbolt, torchdata"
python tests/datasets/graph_tokenizer_dgl_smoke.py
python -m pytest -q -rs \
  tests/transforms/test_graph_tokenizer.py \
  tests/datasets/test_graph_tokenizer_datasets.py \
  tests/models/test_graph_tokenizer_models.py \
  tests/models/test_graph_tokenizer_dataset_metadata.py \
  tests/models/test_graph_tokenizer_pipeline.py
python -m pytest -q -rs tests/transforms/test_graph_tokenizer.py -k cpp
```

最后一条检查已构建的可选 Python/C++ BPE 一致性；扩展缺失会显式 skip。
真实 release bundle 测试需要设置 `GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE`，否则显式 skip，
不能记为真实数据验证通过。下文给出无需真实数据或正式 GTE checkpoint 的 tiny smoke 命令。

### 2026-10-17 CPU 验证记录

在独立 Python 3.10.20 venv 和独立源码副本中，按上述顺序安装；源码验证基线为
`dde429584e993ec3837e97f6f38d7f083f0fb078`（已包含梯度累积修复）。本阶段仅更新
安装文档和 CI，不改变模型、输入协议、数据划分或六组 preset。

- PyTorch `2.1.2+cpu`、torchvision `0.16.2+cpu`、torchaudio `2.1.2+cpu`、DGL `2.1.0`、torchdata `0.7.1`。
- GAMMA Lab TLX nightly revision `8fbe30d0588a5e58673d9cfad77b111af93182bb`；包元数据版本 `0.5.9`，运行时 `tlx.__version__` 为 `0.5.8`。
- NumPy `1.26.4`、setuptools `69.5.1`、wheel `0.48.0`、pybind11 `3.1.0`、ninja `1.13.2`、pytest `9.1.1`；Linux x86_64、GCC/G++ `13.3.0`，`TL_BACKEND=torch`、`DGLBACKEND=pytorch`、`GAMMAGL_WITH_CUDA=0`。
- paper extra editable 安装及原生扩展构建完成；上述依赖和 GraphBolt 均可导入，`pip check` 无依赖冲突，DGL pickle smoke 通过。
- 五个回归测试文件：**253 passed、1 skipped**。唯一 skip 为未配置真实 release bundle；另有一条 PyTorch 使用 `pkg_resources` 的弃用警告。
- 已构建 C++ BPE，对照测试：**2 passed、57 deselected**，同一条弃用警告；没有以扩展缺失的 skip 代替通过。
- tiny BERT 和显式 `--allow-random-gte-init` 的 tiny GTE smoke 均完成；GTE summary 明确为 `pretrained: false`、`reproduction: false`、`mode: random_development`。
- `git diff --check` 通过。CUDA 构建／GPU 运行、真实 release bundle 和正式预训练 GTE checkpoint 初始化未验证；没有重跑六组正式 benchmark。

下载曾遇到官方 CPU CDN 读取超时、GitHub 连接失败和 PyPI 连接中断；最终复用本次获取的
同 revision TLX 源码对象缓存及同版本 wheel 缓存／镜像完成安装。没有用其他 Python／
PyTorch／DGL 组合的测试结果替代本记录，也没有升级或卸载正在训练的环境依赖。

## 数据准备

Mutagenicity、DBLP_v1 和 AQSOL 共用官方 release bundle。启动任意训练 worker 前，先运行一次
单进程准备步骤：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --prepare-data \
  --data-root data
```

它会在同一 `--data-root` 下下载、SHA-256 校验、缓存并 materialize bundle；重复运行复用已校验
缓存。Dataset 构造和训练 worker 不会隐式下载它，因此请勿让多个 worker 并发执行准备步骤。
设置 `GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/path/to/local/bundle` 可使用本地已验证 bundle。
训练时已拟合的 tokenizer 与编码后的 train/val/test records 会缓存到
`DATA_ROOT/.graph_tokenizer_preprocessing_cache/`；同一配置的后续 seed 可复用该缓存，
`--preprocessing-cache-dir PATH` 可改变位置。
缓存键对 Dataset 实际读取的 processed 图数据与 split 文件计算 SHA-256 内容摘要；首次
处理时先读取 raw，待 processed 生成后按 processed 内容写入缓存。因此仅改动 raw 而未重新
生成 processed 时，训练输入与缓存均仍以现有 processed 数据为准。

## 训练与评估

```bash
export TL_BACKEND=torch

python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset mutagenicity --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset mutagenicity --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset dblp --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset dblp --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset aqsol --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset aqsol --encoder gte --data-root data --device cuda
```

默认每个命令运行五个 seeds；可显式指定：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset mutagenicity --encoder bert --data-root data --device cuda \
  --seeds 42 43 44 45 46
```

Mutagenicity 和 DBLP_v1 使用两个 logits 的 CrossEntropy，并在每个图的多次 serialization
logits 平均后取 `argmax` 计算 Accuracy。AQSOL 使用 MSE 训练；其标签归一化只用 train split
中每个唯一 graph_id 的标签统计一次，评估将预测和标签恢复到原始尺度后计算 MAE。
标签归一化使用训练标签的总体标准差（除以训练图数），与五 seed 指标汇总的样本标准差
（`ddof=1`）区分。

常用覆盖参数包括 `--pretrain-batch-size`、`--finetune-batch-size`、
`--pretrain-learning-rate`、`--learning-rate`、
`--pretrain-epochs`、`--finetune-epochs`、`--max-length`、`--bpe-merges`、
`--bpe-backend`、`--num-serializations`、`--early-stopping-patience`、
`--pretrain-weight-decay`、`--finetune-weight-decay`、`--gradient-accumulation-steps` 与 `--seeds`。
使用 `--gte-checkpoint-cache-dir PATH` 指定 GTE checkpoint 的 Hugging Face 缓存。

每次运行将 checkpoint 与 summary 写入 `run_01_seed_42/`、`run_02_seed_43/` 等目录；父级
`summary.json` 保存聚合结果。early stopping 的方向由 dataset metadata 驱动：Accuracy 取最大值，
MAE 取最小值。
当前 summary 会保存 seed、指标、阶段 history、checkpoint 路径及 GTE 初始化信息；不会自动
保存 Git commit、dirty diff、完整 CLI、生效配置或数据／split 内容摘要。正式记录需要另行
归档这些信息，并绑定该次 tokenizer、缓存及 checkpoint。这些是记录归档要求，
不代表当前 trainer 已实现自动归档。

## 极小 smoke 测试

```bash
export TL_BACKEND=torch
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --smoke --dataset aqsol --encoder bert --device cpu \
  --seeds 42 \
  --output-dir /tmp/graph-tokenizer-bert-smoke

python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --smoke --dataset aqsol --encoder gte --allow-random-gte-init --device cpu \
  --seeds 42 \
  --output-dir /tmp/graph-tokenizer-gte-smoke
```

BERT 的 `--smoke` 使用 tiny architecture。GTE 的 tiny smoke 必须显式加入
`--allow-random-gte-init`；该随机初始化模式仅用于开发／pipeline 检查，不能作为正式
reproduced result。

### TLX 后端与训练接口

训练入口支持 `TL_BACKEND=torch` 和 `TL_BACKEND=tensorflow` 的 eager 模式。
上面的两条 tiny 命令也可在安装 GAMMA Lab TLX nightly 和 TensorFlow 的独立环境中，
设置 `TL_BACKEND=tensorflow` 后运行；普通 BERT／随机 GTE 训练不需要安装 PyTorch、
DGL 或 Hugging Face 模型。真实 release 数据的 DGL pickle 读取和正式 GTE checkpoint
下载仍有各自的可选依赖。这里未声明 Paddle、MindSpore、Jittor 或 TensorFlow graph 模式
的训练支持，也未验证 GPU。

模型计算继续使用原生 TLX 层；数据使用 TLX DataLoader，损失使用 TLX losses，
优化使用 TLX Adam 加独立 weight decay，保留此前 AdamW 的更新语义。
损失通过 `WithLoss` 组织。当前 GAMMA TLX nightly 的 `WithGrad` 没有 torch 分支，
`TrainOneStep` 会逐 microbatch 立即更新，torch 的 `ClipByGlobalNorm` 未实现，
因此 `tlx_training.py` 在这些边界单独适配：torch 使用 TLX optimizer 的梯度接口，
TensorFlow 使用与 TLX `TrainOneStep` 相同的 eager `GradientTape`；累积后的梯度按目标数
归一化，检查有限性，再用 TLX ops 计算全局范数和裁剪，最后调用 TLX optimizer 更新。
torch optimizer 对 `parameter.grad` 的读取也局限在该适配中。

torch 最佳 checkpoint 保持原有 `.pt` 格式；TensorFlow 使用不含 pickle 的 `.npz`
权重和 schema／epoch／score metadata，并通过 TLX 恢复权重。两种 checkpoint 不跨后端互载。
成功结果仍使用相同 JSON 字段，非有限训练／评估结果会使运行失败。

公共 CI 继续使用原有 torch 后端，负责核心安装与公共测试；新增 paper 环境在独立 job
安装固定版本及 `graph-tokenizer-paper` extra、运行 DGL smoke 与 GraphTokenizer 回归。
两个 job 相互独立，公共检查不以 paper extra 安装成功为前提。

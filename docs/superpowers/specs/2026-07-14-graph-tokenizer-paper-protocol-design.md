# GraphTokenizer Paper-Protocol Design

## Goal

在不破坏 GammaGL 多后端公共接口的前提下，为 QM9、OGBG-molhiv 和
Peptides-struct 提供仅支持 `TL_BACKEND=torch` 的严格论文复现实验路径。

## Constraints

- GammaGL 原有 TensorLayerX 模型、`Graph`、`Dataset` 和其他后端必须保持可导入。
- 论文路径必须使用 Torch、HuggingFace Transformers 和 CUDA；这些依赖不能成为
  `gammagl.models` 的全局必需依赖。
- 数据必须使用官方来源及官方固定划分；缺字段、错误维度、索引越界、划分交叉或
  不完整时必须失败，绝不能补零、截断或静默过滤。
- 论文路径的结果必须记录数据来源/哈希、配置、Git revision、GPU/CUDA、随机种子和
  完整 checkpoint 状态。
- 不修改 GammaGL 的基础图数据结构和现有多后端训练模块。

## Architecture

保留 `gammagl/transforms` 中与图结构相关的纯 Python/TensorLayerX 实现，并补齐可逆
序列化、严格 token/vocabulary 校验和论文所需的序列化选择。训练入口新增
`--protocol paper`，此模式通过延迟导入的 Torch/HuggingFace 适配器运行；默认
`--protocol gammagl` 仍使用现有 TensorLayerX 模型。

论文模型适配器位于 `examples/graph_tokenizer/official_model_adapter.py`，不会被
`gammagl.models.__init__` 导入。它提供统一的 BERT/GTE 编码器、MLM 头和任务头：BERT
从 `BertModel(BertConfig)` 随机初始化，GTE 从
`Alibaba-NLP/gte-multilingual-base` 加载并调整输入词表。两者都支持 `mean` 和 `cls`
pooling，并在 Torch tensor 上训练。

## Data And Tokenization Contract

每个数据集的 GammaGL loader 负责下载或确定性转换为统一的原始契约：
`data.pkl(.gz)`、`train_index.json`、`val_index.json` 和 `test_index.json`。每个样本必须
提供完整节点特征、边特征、边索引和精确标签维度。转换器必须严格复刻论文的数据集
特征规则：分子图提取原子序数和键类型，并分别映射到固定的奇数和偶数 token 域；它
不能采用“取第一列”的无约束通用规则，也不能自行补零或截断。

处理阶段只在训练划分拟合频率统计和 BPE merge rules，并写出序列化、词表、tokenizer
配置与数据哈希。特殊 token 固定为 `PAD=0`、`UNK=1`、`MASK=2`、`CLS=3`、`SEP=4`、
`NODE_START=5`、`NODE_END=6`、`COMPONENT_SEP=7`。序列化必须可解码回原始有标签图；
对固定确定性方法，节点重编号不应改变最终 token 序列。

QM9 采用论文的 16 个属性及逐 `target_property` 微调。OGBG-molhiv 使用官方 split 和
OGB ROC-AUC evaluator。Peptides-struct 使用官方 split 与逐任务 MAE 的平均值。任何缺失
标签按对应官方 evaluator 的 NaN 语义处理，不能转换为零。

## Training Protocol

`paper` 模式要求 CUDA、`torch`、`transformers`、`ogb` 和可用的 C++ BPE 后端；启动时会
给出精确缺失项。训练使用论文配置：随机种子 42 起连续五次运行、BPE 2000 merges、MLM
mask probability 0.09、AdamW、预训练/微调各 200 epochs、warmup、梯度裁剪和验证集早停。
每个数据集的 paper preset 可以覆盖学习率、batch size、序列化方法及最大长度。

训练只在 train 上拟合 tokenizer 和更新参数。每个 epoch 只评估 validation；恢复验证最佳
checkpoint 后，对 test 仅评估一次。checkpoint 保存模型、优化器、scheduler、tokenizer、
当前 epoch、最佳指标和全部 RNG 状态。

## Files

- Modify: `gammagl/datasets/_molecular_benchmark.py`, `qm9.py`, `ogbg_molhiv.py`,
  `peptides_struct.py` and dataset tests for strict source/schema/split handling.
- Modify: `gammagl/transforms/graph_serializer.py`, `graph_bpe.py`,
  `graph_tokenizer.py` and transform tests for reversible deterministic tokenization.
- Create: `examples/graph_tokenizer/official_model_adapter.py` and Torch-only tests.
- Modify: `examples/graph_tokenizer/graph_tokenizer_trainer.py` for protocol selection,
  paper preprocessing, training, evaluation, checkpointing and manifest generation.
- Create: `examples/graph_tokenizer/configs/qm9_paper.json`,
  `molhiv_paper.json`, `peptides_struct_paper.json`.
- Modify: `third_party/graph_bpe_cpp` packaging/import code and add a Linux build/import test.
- Modify: `examples/graph_tokenizer/README.md` with the paper-protocol prerequisites and
  remote execution commands.

## Error Handling And Validation

Preflight unpickles raw data, validates graph schemas, labels, split ranges/disjointness and
expected counts, verifies processed artifact hashes, checks the C++ backend and validates the
Torch/HuggingFace/CUDA requirements in `paper` mode. A mismatch aborts before training.

## Test Strategy

Tests cover official-like fixture conversion, schema failures, split failures, tokenizer
round trips, node-permutation invariance, BPE vocabulary limits, OGB evaluator parity, QM9
target selection/normalization, checkpoint restoration and a one-epoch Torch paper-mode CLI
smoke test. Linux CI additionally builds and imports the C++ BPE extension.

## Non-Goals

This work does not alter GammaGL's generic dataset abstractions, message-passing layers or
non-Torch backends. The existing TensorLayerX graph-tokenizer path remains available but is
not presented as a paper-reproduction result.

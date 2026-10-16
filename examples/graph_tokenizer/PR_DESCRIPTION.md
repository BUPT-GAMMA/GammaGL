## 概述

在 GammaGL 中集成 GraphTokenizer 的图序列化、Graph BPE、图 token 构建、MLM 预训练和
下游图分类／回归流程。统一入口为 `examples/graph_tokenizer/graph_tokenizer_trainer.py`，
支持 Mutagenicity、DBLP_v1 和 AQSOL，以及原生 TensorLayerX `GraphBERT`／`GraphGTE`。
模型前向使用 TLX；正式 GTE 通过 HF-to-TLX converter 初始化固定来源的预训练编码器。

## 当前输入与训练协议

- Feuler 面向简单无向图，使用遍历首次访问的连续局部节点引用；原始编号保留在
  `local_to_original` provenance metadata。支持非连通图、孤立节点、空图和显式 `start_node`。
- `x=None` 使用统一零节点标签，恢复后的 `x` 也是默认标签；显式整数标签按输入保留。
  benchmark Dataset 要求提供节点特征。
- BPE 和回归标签统计只在训练集拟合。AQSOL 每个唯一训练 `graph_id` 只统计一次，
  修复按 serialization 次数加权的问题；评估在原标签尺度计算 MAE。
- 默认 seeds 为 `42 43 44 45 46`。MLM 后进行微调，以验证集选择最佳 checkpoint，恢复后
  进行最终测试。同图各次 serialization 的 logits／预测先平均，再计算 Accuracy／MAE。
- 长度包含 special tokens，超过 `max_length` 会报错；不截断图结构。
- 预处理缓存对实际读取的 processed 图数据和 split 计算 SHA-256，并纳入输入协议及
  预处理配置。首次生成 processed 后按其内容写入，后续 seed 可复用，保留原子写入。

当前 serializer protocol／tokenizer schema 为 `4 / 4`，预处理缓存格式为 `5`。
旧 protocol 2 的原始编号引用已由局部引用替代，后续又修复了无特征图的编号标签和缓存来源跟踪。

## 六组当前复现实验结果

按本次提供的实验确认，六组指标继续作为最新版代码的五 seed 复现实验成绩保留。
`mean ± std` 的 std 为跨 seed 样本标准差（`ddof=1`）。Accuracy 均值为百分数，std 单位为
百分点，两者均由 trainer 的 0–1 输出乘以 100；AQSOL 为原尺度 MAE。

| 数据集 | Encoder | 指标 | 论文／官方参考 | 当前 GammaGL reproduction |
| :-- | :-- | :-- | :-- | :-- |
| Mutagenicity | BERT | Accuracy (%) ↑ | 87.5 ± 0.9 | 76.73 ± 1.75 |
| Mutagenicity | GTE | Accuracy (%) ↑ | 90.1 ± 0.7 | 68.11 ± 5.88 |
| DBLP_v1 | BERT | Accuracy (%) ↑ | 93.2 ± 0.1 | 92.25 ± 1.28 |
| DBLP_v1 | GTE | Accuracy (%) ↑ | 93.6 ± 0.1 | 92.14 ± 0.45 |
| AQSOL | BERT | 原尺度 MAE ↓ | 0.648 ± 0.008 | 0.7395 ± 0.0072 |
| AQSOL | GTE | 原尺度 MAE ↓ | 0.609 ± 0.016 | 0.7077 ± 0.0069 |

论文参考栏与 GammaGL 实验记录独立。六组现存本地五 seed summary、汇总和日志输出相符；
重新计算 mean／sample std 后，均与表中显示精度一致。

## 实验配置与可追溯性

当前六组 `PAPER_CONFIGS` 使用官方 release 的固定 train／val／test split、训练集拟合的
BPE（2000 merges、最小频率 2）、100 次序列化、最终长度上限 8096。BERT 为
512 隐藏维／4 层／4 头／2048 FFN；GTE 为 768／12／12／3072，使用 RoPE 和 8192 位置容量。
完整默认训练参数及 CLI 覆盖规则见 `examples/graph_tokenizer/README.md`。

实跑配置与当前默认 preset 分别记录：匹配数值的本地 retry 日志记录 AQSOL BERT batch
`8/8`、梯度累积 `4`，三组 GTE batch `2/2`、梯度累积 `16`；两组分类 BERT 缺少完整 CLI。
来源路径、各 seed 原精度指标、文件 SHA-256 及证据限制见
`examples/graph_tokenizer/EXPERIMENT_AUDIT.md` 和
`examples/graph_tokenizer/experiment_records/audit_2026-10-16.json`。

最新版运行的精确 commit／代码快照与协议归属尚不能独立核实：数值匹配的本地记录所引用的
现存缓存为 schema 2，checkpoint 缺少 schema 字段，AQSOL 缓存统计量也不符合当前唯一图
归一化规则。这是与当前 schema 4 的具体产物冲突，不能拿这些文件充当最新版重跑的版本凭据。
当前成绩按本次确认保留；最新运行的完整配置、运行时数据／split 摘要及 tokenizer／checkpoint
绑定记录仍需补齐。summary 当前不自动保存全部这些信息；外部本地路径也不是仓库附带的日志。

## 产物兼容性

- 旧／缺失 schema 的 tokenizer 在编码或解码时被拒绝，完整 result 也检查 protocol／schema；
  裸整数 token 流没有版本 metadata，不能自动识别旧来源，应从原图重新生成。
- 缓存检查格式 5 和 tokenizer schema 4，旧键或旧格式不复用；没有自动迁移。
- 下游 checkpoint 在载入模型权重前检查 schema 4。缺少或不匹配会报错；该检查不验证具体
  tokenizer 词表或数据指纹，同 schema 仍需配套产物，相同 embedding 形状不能作为兼容依据。
- 独立 BPE codebook 的加载不检查输入协议，也不等于完整 tokenizer 的持久化。
- 正式 GTE 来源为 `Alibaba-NLP/gte-multilingual-base` revision
  `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`，checkpoint SHA-256 为
  `f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`；校验官方配置、
  tensor shape 和 135/135 encoder 映射。图词表 embedding 与任务头单独初始化。
  `--allow-random-gte-init` 标记为 `reproduction: false`，仅用于开发。

## 验证与限制

本阶段执行以下相关回归（实际 TLX torch 后端、本地小型 fixture）：

```bash
TL_BACKEND=torch python -m pytest -q -rs tests/transforms/test_graph_tokenizer.py tests/datasets/test_graph_tokenizer_datasets.py tests/models/test_graph_tokenizer_models.py tests/models/test_graph_tokenizer_dataset_metadata.py tests/models/test_graph_tokenizer_pipeline.py
```

结果为 **209 passed, 1 skipped**。跳过项是未配置
`GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE` 的真实 release bundle smoke。
另以 Python 标准库逐组复算 30 个 seed 指标、核对来源文件 SHA-256，并检查 README 与 PR 描述草稿
的六行结果表完全一致；`git diff --check` 通过。

六组完整训练、真实 bundle／正式 GTE 全权重端到端实验及全仓库测试未运行：本阶段限定为
文档与记录审计，运行了相关回归，没有下载数据或权重。现存 GTE 初始化 manifest 和
checkpoint metadata 的只读检查不等同于重新执行正式 GTE 初始化／训练。

不支持有向图语义、自环或并行边；结构签名或组件排序打平时仍可能受原始编号影响，
因此不保证任意图的全局置换不变性。完整结果保留 lossless serialized tokens；含未见标签的
模型 `input_ids` 可能含 `UNK`，不能单独逆转为完整图。轻量测试和数值审计不替代完整训练，
也不补足缺少的最新版实验归属记录。

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

GammaGL reproduction 为五个 seed 的最终测试结果，报告 `mean ± std`（样本标准差，`ddof=1`）。

| 数据集 | Encoder | 论文／官方结果 | GammaGL reproduction |
| :-- | :-- | :-- | :-- |
| Mutagenicity | BERT | 87.5 ± 0.9 Accuracy | 0.7673 ± 0.0175 |
| Mutagenicity | GTE | 90.1 ± 0.7 Accuracy | 0.6811 ± 0.0588 |
| DBLP_v1 | BERT | 93.2 ± 0.1 Accuracy | 0.9225 ± 0.0128 |
| DBLP_v1 | GTE | 93.6 ± 0.1 Accuracy | 0.9214 ± 0.0045 |
| AQSOL | BERT | 0.648 ± 0.008 MAE | 0.7395 ± 0.0072 |
| AQSOL | GTE | 0.609 ± 0.016 MAE | 0.7077 ± 0.0069 |

Trainer 接受 `mutag` 和 `muta` 作为 Mutagenicity 的 aliases。`dblp` 对应
**DBLP_v1 graph-classification** benchmark，数据集类为 `GraphTokenizerDBLP`。

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

## 固定的六组实验 preset

`PAPER_CONFIGS` 基于固定的
[GraphTokenizer 官方源码 revision
`98343a6b025a48fbb6859cd812a12b81ec3ac3cc`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)
整理为六组 benchmark presets。训练使用统一的 serializer/tokenization pipeline，默认使用
Python BPE，`cpp` 是可选 backend。

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

`--gradient-accumulation-steps` 仍是通用 CLI 功能；每个完整或最后的 partial accumulation
window 都使用其实际 window size 作为相同 divisor，optimizer 与 scheduler 的 step 数保持一致。

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

正式论文运行时固定使用 Python 3.10、PyTorch 2.1.2、DGL 2.1.0 和
torchdata 0.7.1。DGL 2.1 的 GraphBolt 动态库按 PyTorch 版本构建，不能将这里的
PyTorch 升级到其他版本。先按 CPU 或 CUDA 环境安装对应的 PyTorch 2.1.2 wheel，再安装 extra；
例如 CPU 环境：

```bash
python --version  # Python 3.10.x
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[graph-tokenizer-paper]"
```

该 extra 同时固定 DGL／torchdata 兼容组合，并包含 `huggingface-hub`、`safetensors`；
正式 GTE 初始化需要它们来获取并转换固定 checkpoint。

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

常用覆盖参数包括 `--pretrain-batch-size`、`--finetune-batch-size`、
`--pretrain-learning-rate`、`--learning-rate`、
`--pretrain-epochs`、`--finetune-epochs`、`--max-length`、`--bpe-merges`、
`--bpe-backend`、`--num-serializations`、`--early-stopping-patience`、
`--pretrain-weight-decay`、`--finetune-weight-decay`、`--gradient-accumulation-steps` 与 `--seeds`。
使用 `--gte-checkpoint-cache-dir PATH` 指定 GTE checkpoint 的 Hugging Face 缓存。

每次运行将 checkpoint 与 summary 写入 `run_01_seed_42/`、`run_02_seed_43/` 等目录；父级
`summary.json` 保存聚合结果。early stopping 的方向由 dataset metadata 驱动：Accuracy 取最大值，
MAE 取最小值。

## 极小 smoke 测试

```bash
export TL_BACKEND=torch
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --smoke --dataset aqsol --encoder bert --device cpu \
  --seeds 42 \
  --output-dir /tmp/graph-tokenizer-smoke
```

BERT 的 `--smoke` 使用 tiny architecture。GTE 的 tiny smoke 必须显式加入
`--allow-random-gte-init`；该随机初始化模式仅用于开发／pipeline 检查，不能作为正式
reproduced result。

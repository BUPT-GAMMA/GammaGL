# GraphTokenizer

本示例为 GraphTokenizer 提供图序列化、Graph BPE 词元化、掩码语言模型预训练、监督微调、检查点保存和多随机种子评估功能。

## 论文

GraphTokenizer 由论文[连接图与 Transformer 的图词元化](https://openreview.net/forum?id=jCctxI1BGF)（ICLR 2026）提出。

论文实验协议固定到官方实现提交
`98343a6b025a48fbb6859cd812a12b81ec3ac3cc`。它采用频率引导的欧拉序列化
（Feuler），每图请求 100 个起点变体（不超过节点数），仅在训练集划分上拟合
Graph BPE。训练和验证按图随机取一个变体，测试对同一图的全部变体预测取平均；
最佳检查点只由验证集选择，最终报告五次运行的均值和样本标准差（`ddof=1`）。

预训练在 BPE 前使用 RandomSwap；微调在 BPE 前使用 RandomSwap 和
SequenceMasking，并以 0.3 概率向池化特征加入标准差 0.01 的高斯噪声。学习率
采用线性 warmup 后余弦衰减，最低为初始学习率的 1%。BERT 的位置嵌入表为
8096，但与官方数据管线相同，实际输入上限固定为 768；GTE 的模型容量为 8192，
论文训练管线的实际输入上限为 8096。

Feuler 序列化器在其支持的无向简单图范围内是可逆的。论文中的 GTE
实验会将版本固定的官方编码器加载到原生 TLX GraphGTE 实现中；图词元嵌入和
任务头均采用全新初始化。

Feuler 仅接受无向简单图：不支持自环和平行边，也不接受显式声明
`directed=True` 或 `is_directed()=True` 的图对象。COO 可以只存储每条无向边
一次，也可以采用对称形式存储；两种表示都会被规范化为相同的无向边。原始的
单向 COO 无法表达独立的有向图语义，因此只会被解释为受支持的无向图存储形式。

## 数据集

以下论文实验命令覆盖三个数据集：

- QM9：只预测论文默认的 HOMO；训练标签按训练集 z-score 标准化，最终将预测
  反变换为原始 eV 后计算 MAE。
- OGBG-molhiv：双 logit、无类别加权的交叉熵，使用 OGB ROC-AUC 报告结果。
- Peptides-struct：11 目标按训练集逐目标 z-score 标准化，以 L1 训练；评估时
  反变换到原始标签空间，逐目标计算 MAE 后等权平均。

Peptides-struct 的 MLM 阶段使用与官方 `peptides-func` 相同的肽图语料；两套任务
的图与划分相同，MLM 不读取下游标签，因此本实现直接复用 Peptides-struct 中等价
的图数据，并在运行清单中记录 `pretraining_graph_corpus=peptides-func`。

GammaGL 数据集类会在首次使用时下载 GraphTokenizer 官方发布的数据包。下载的
归档文件会在解压前使用官方公布的 SHA-256 进行校验：

```
5c437c3c0d4b7278379c0e70d57f98148e5c815d753d8cf68e2a45952bcce459
```

数据包会缓存在 `<data-root>/.graph_tokenizer_release` 下，然后复制到各数据集的
常规 `raw/` 目录。通过
`GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/path/to/bundle` 指定的本地归档文件也会
使用相同的摘要进行校验。将该变量设置为目录表示显式启用本地开发覆盖；程序
不会从用户提供的 URL 下载文件。只有自动下载的远程数据包通过校验后，程序才会
反序列化其中发布的 `data.pkl(.gz)` 文件。

## 环境要求

论文模式要求：

- `TL_BACKEND=torch`
- PyTorch 2.1.2，配套 CUDA 12.1
- DGL 2.4.0，配套 CUDA 12.1
- PyTorch Geometric 2.4.0
- TensorLayerX、NumPy、OGB 和 GammaGL 软件包
- 用于固定版本原生 TLX GTE 检查点转换的 `huggingface-hub` 和 `safetensors`
- 执行以下命令所需的原生 Graph BPE 扩展

使用以下方式安装依赖，不会将论文专用软件包加入 GammaGL 的核心依赖集合：

```bash
pip install -e '.[graph-tokenizer-paper]'
# 也可以在版本固定的论文环境中使用 examples/graph_tokenizer/requirements.txt。
```

`transformers` 是可选依赖：它仅用于检查点和参考实现等价性测试，不会用于正式的
TLX GraphBERT/GraphGTE 训练前向传播路径。

在仓库根目录编译原生 Graph BPE 后端：

```bash
export TL_BACKEND=torch
export DGLBACKEND=pytorch
python third_party/graph_bpe_cpp/setup.py build_ext --inplace
```

论文模型直接使用 TensorLayerX 的公共层和运算；GraphBERT 和 GraphGTE 不依赖 Hugging Face Transformers。

## 运行方法

所有论文超参数均通过 `argparse` 显式传入。以下命令使用相对于仓库的数据、缓存和结果路径，可以直接在 GammaGL 仓库根目录运行。

### QM9 + BERT

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --protocol paper \
    --dataset qm9 \
    --target-property homo \
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
    --max-position-embeddings 8096 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --min-frequency 2 \
    --num-realizations 100 \
    --aggregation-mode avg \
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
    --target-property homo \
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
    --min-frequency 2 \
    --num-realizations 100 \
    --aggregation-mode avg \
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
    --max-position-embeddings 8096 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --min-frequency 2 \
    --num-realizations 100 \
    --aggregation-mode avg \
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
    --min-frequency 2 \
    --num-realizations 100 \
    --aggregation-mode avg \
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
    --max-position-embeddings 8096 \
    --data-root data \
    --bpe-backend cpp \
    --paper-cache-root cache/graph_tokenizer \
    --num-merges 2000 \
    --min-frequency 2 \
    --num-realizations 100 \
    --aggregation-mode avg \
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
    --min-frequency 2 \
    --num-realizations 100 \
    --aggregation-mode avg \
    --runs 5 \
    --seed 42 \
    --paper-amp off \
    --no-paper-tf32 \
    --output-dir logs/graph_tokenizer/peptides_struct_gte
```

在开始长时间运行前，可以向相应命令添加 `--preflight`，在不训练的情况下检查数据集、模型、运行环境和 BPE 后端。`--resume` 会从 `last_state.pt` 恢复最近的训练阶段、优化器、调度器、随机数生成器和 AMP 缩放器状态；`best.pt` 仍然是轻量级的最佳模型检查点。

执行一个非论文模式的小型冒烟测试：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
    --smoke \
    --dataset qm9 \
    --model bert \
    --data-root data \
    --bpe-backend python
```

## 实验结果

论文报告的五次运行平均结果如下：

| 数据集 | 编码器 | 指标 | 论文结果 | GammaGL 状态 |
| --- | --- | --- | ---: | --- |
| QM9 | BERT | 原始尺度 MAE ↓ | 0.122 | 需要重新验证 |
| QM9 | GTE | 原始尺度 MAE ↓ | 0.071 | 需要使用官方权重重新验证 |
| OGBG-molhiv | BERT | ROC-AUC ↑ | 82.6% | 需要重新验证 |
| OGBG-molhiv | GTE | ROC-AUC ↑ | 87.4% | 需要使用官方权重重新验证 |
| Peptides-struct | BERT | 平均 MAE ↓ | 0.247 | 需要重新验证 |
| Peptides-struct | GTE | 平均 MAE ↓ | 0.242 | 需要使用官方权重重新验证 |

“论文结果”是 GraphTokenizer 论文中报告的数值。在严格运行记录官方 GTE 检查点来源并以原始标签单位报告 QM9 结果之前，不得将 GammaGL 结果与论文结果进行比较，也不得称其已经复现。

每次运行都会在 `--output-dir` 下保存结果文件，包括汇总 JSON、逐次运行 CSV、Markdown/LaTeX 论文表格、运行时清单、检查点和论文协议状态。JSON 仅作为输出格式，不用于输入参数配置。

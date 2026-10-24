# GraphTokenizer 训练

`graph_tokenizer_trainer.py` 是训练入口。它使用 GammaGL 的 Eulerian／Feuler 序列化器、Graph BPE、
tokenizer、数据集以及原生 TensorLayerX `GraphBERT` / `GraphGTE`。

正式路径默认以种子 `42 43 44 45 46` 完成五次彼此独立的完整训练：按官方预处理入口，
以 `train → val → test` 的顺序使用全部图结构／token 拟合频率、BPE 和词表（`--fit-scope all`）。
标签归一化只用训练标签，在完整训练集上进行 MLM，按 MLM 验证 loss 早停并恢复最佳预训练
checkpoint，再微调、验证、恢复最佳微调 checkpoint 并最终测试。
最终指标为五次最终测试的 `mean ± std`（样本标准差，`ddof=1`）。测试集只会在恢复
最佳 checkpoint 后评估；预处理会使用验证／测试图结构，因此属于 transductive 协议。

`--fit-scope train` 是显式的 inductive 变体，仅用训练图拟合 tokenizer，不属于同组官方
复现实验。它的 checkpoint 和 summary 独立写入 `<output-dir>/<dataset>/<encoder>/train_only/`；
默认 all 路径仍为 `<output-dir>/<dataset>/<encoder>/`，两种范围不能混合汇总。

长度超过 `max_length` 的序列化图会被拒绝而非截断，因为截断会丢失图结构。CLI 参数会覆盖
`PAPER_CONFIGS` 中对应 preset 的值。

## Benchmark 测试结果

以下为最新报告的测试结果，按 `mean ± std` 展示；论文／官方参考独立列出。

| 数据集 | Encoder | 指标 | 论文／官方参考 | 最新测试结果 |
| :-- | :-- | :-- | :-- | :-- |
| Mutagenicity | BERT | Accuracy (%) ↑ | 87.5 ± 0.9 | 83.4701 ± 0.7002 |
| Mutagenicity | GTE | Accuracy (%) ↑ | 90.1 ± 0.7 | 83.9249 ± 0.6304 |
| DBLP_v1 | BERT | Accuracy (%) ↑ | 93.2 ± 0.1 | 92.2405 ± 0.7432 |
| DBLP_v1 | GTE | Accuracy (%) ↑ | 93.6 ± 0.1 | 92.9372 ± 0.9202 |
| AQSOL | BERT | 原尺度 MAE ↓ | 0.648 ± 0.008 | 0.6410 ± 0.0048 |
| AQSOL | GTE | 原尺度 MAE ↓ | 0.609 ± 0.016 | 0.6361 ± 0.0231 |

Accuracy 的均值用百分数表示，`±` 后的标准差单位为**百分点**；相对于 trainer 输出的
0–1 Accuracy，均值和标准差均乘以 100。最新测试结果保留四位小数。
AQSOL 的 MAE 保持原标签尺度，不做百分数转换。

### 实验记录与可追溯性

正式运行应保存精确 commit／源码摘要、完整配置、五个 seed 的原始指标、完整日志、
配套预处理产物及两阶段 checkpoint。上表记录报告的汇总成绩，配套运行产物应独立归档；
smoke 和单元测试用于功能验证。

Trainer 接受 `mutag` 和 `muta` 作为 **Mutagenicity** 的 aliases，不指 188-graph `MUTAG`。
`dblp` 对应 **DBLP_v1 graph-classification** benchmark，数据集类为 `GraphTokenizerDBLP`，
与异构节点分类数据集 `DBLP` 区分。

## 输入协议与支持的图范围

正式默认使用 `official` 协议，具体 serializer 由六组 preset 显式选择。
它复现固定官方源码的实际 token 流，不添加节点引用，不进行结构签名排序，也不保证
任意节点重编号后的序列相同。它支持标量标签的简单无向图、非连通图及孤立节点，拒绝
空图、显式有向语义、自环及并行边。COO 弧按输入保留；正式数据应保留 release 的双向弧，
单向 COO 不保证与双向 COO 等价。Eulerian 在官方度数条件下复制弧，随后若路径需要
不存在的输入边，会明确报错，不静默生成错误结构。

Feuler 使用定向 triplet 频率、float32 log10 权重及官方编号排序。TU 批量与 singleton
getter 分别按官方规则编码；AQSOL 使用原子序数和键类型，保留官方常见边类型省略、
组件排序及普通 token 7 分隔规则。BPE 在原始完整序列上学习，首个 merge ID 为 max(base)+2；
训练增强先对原始 token 做窗口 swap／mask，再 BPE 和词表编码。

GammaGL 普通 token 按 ID 排序，官方按词频排序，通过完整词表双射对照模型输入。
保留普通 raw-0／PAD alias；未见原始 token 使用独立 OOV sentinel 后映射 UNK，避免冒充 merge ID。
词表双射绑定具体拟合语料；独立 GraphBPE 的 `vocab_size` 表示 ID 上界。
超长输入明确拒绝，增强后超长回退原输入；官方词表编码器会截断。

官方流没有节点身份，也不保留所有边 token；`deserialize()`／`restore_input_metadata()`／
`decode_graph()` 对该协议抛出 `NotImplementedError`，不保证图 round trip。
逐项对照依据见[官方离线参考与再生成](#官方离线参考与再生成)。

### 独立可选的可逆协议

仅显式 `--token-protocol local_refs` 才会在 trainer 中使用 GammaGL 可逆引用协议。
API 对应 `FrequencyGuidedEulerianSerializer(protocol="local_refs")` 或
`EulerianSerializer(protocol="local_refs")`。该协议以首次访问顺序分配连续局部引用，
重复访问复用引用，跨组件共享编号；裸 token 可恢复同构图，完整 result 的
`local_to_original` 可恢复原始编号。缺少 `x` 时以全零标签替代，不能恢复不存在的特征；
模型输入中的 UNK 不可逆，完整 result 仍保留原序列。该协议不提供任意图的全局同构
规范化，结构签名平局时仍可能依赖节点编号。现有 round-trip、置换示例及旧协议拒绝测试
均显式运行这个协议，不能作为官方协议可逆性的证据。

### 产物兼容性

当前 serializer protocol 为 **5**、tokenizer schema 为 **7**，预处理缓存格式为 **9**。
这些版本用于校验预处理产物与当前代码的兼容性。

| 产物 | 实际检查 |
| :-- | :-- |
| 已拟合 tokenizer | 编码／解码前检查 schema、拟合原始 token 集及拟合时配置；协议／serializer／范围改动须重新 fit。 |
| serializer／tokenizer result | 检查协议／schema；可逆结果还检查协议名称及 method。官方结果不可反序列化。 |
| 预处理缓存 | 键包含协议、serializer、dataset、BPE 模式、fit_scope、版本、数据／split 内容及预处理参数；payload 再检查配置、tokenizer 实际范围及指纹，错配时重新构建。 |
| 下游 checkpoint | 正式 trainer 绑定具体 tokenizer 指纹和 fit_scope；恢复权重前检查 schema、范围和指纹，拒绝跨协议、serializer、BPE、词表或拟合统计复用。 |

指纹涵盖 serializer 配置与官方 revision、频率统计、BPE 规则、原始 token 集、词表、
special IDs、编码模式、模型到官方词表的编号双射及拟合范围、图 ID／变体数。
不自动迁移旧 tokenizer、缓存或下游 checkpoint。

BERT 模型协议为 `model_protocol_version=2`：词嵌入和 Linear 使用普通正态
`N(0, 0.02)`，MLM head 带独立可训练零初始化 bias，PAD 词嵌入行清零且无查表梯度。
位置／类型嵌入继续正常相加，包括 attention_mask=1 的普通 raw-0／PAD alias。
manifest、checkpoint 和 seed／组 summary 记录模型协议；恢复及汇总检查版本。
缺少该版本或版本不同的旧 BERT checkpoint 需使用原代码推理，不能自动补 bias、
清零旧 PAD 权重后当作新模型。GTE 模型及其旧导出兼容规则不变。

`GraphBPE.load_codebook()` 仅加载独立压缩规则，不代表可配套替换 tokenizer；正式
checkpoint 的指纹来自实际已拟合 tokenizer。直接调用低层 checkpoint helper 的无绑定
模型不提供这项训练输入配套保证。正式 GTE 来源 checkpoint 是独立的编码器初始化，
其 revision／SHA-256／架构／shape／覆盖率校验保持原有行为。

## 固定的六组实验 preset

`PAPER_CONFIGS` 基于固定的
[GraphTokenizer 官方源码 revision
`98343a6b025a48fbb6859cd812a12b81ec3ac3cc`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)
整理为六组 benchmark presets。默认使用 Python BPE，`cpp` 是可选 backend。
下表描述当前代码默认配置，不能替代每次运行的完整生效配置及配套记录。

| 实验 | Serializer | 实际 BPE 编码 | 官方配置来源 |
| :-- | :-- | :-- | :-- |
| Mutagenicity + BERT | feuler | all | default_config.yml |
| Mutagenicity + GTE | feuler | none（raw） | default + config/mutagenicity/feuler/gte/config.json |
| DBLP_v1 + BERT | feuler | all | default_config.yml |
| DBLP_v1 + GTE | eulerian | none（raw） | default + config/dblp/eulerian/gte/config.json |
| AQSOL + BERT | feuler | all | default_config.yml |
| AQSOL + GTE | feuler | all | default_config.yml，encoder 切换到 GTE |

固定 revision 没有其余四组的专属 JSON；它们继承 default 的 serializer／BPE 模式。
100 次序列化是这里显式启用的多起点配置，default YAML 的 multiple_sampling.enabled 本身为 false。
`--serializer`、`--token-protocol`、`--bpe-encode-mode` 可覆盖 preset，并参与真实构造和缓存键。
`none` 仍拟合 BPE 规则并构建覆盖全部规则的词表，仅跳过送入模型前的 BPE 编码。
MLM 与下游验证均按官方配置，每轮每图随机选择一个序列变体，不启用训练增强。
分类验证对所选变体做 softmax／argmax，用于最佳微调 checkpoint 选择；最终测试
评估全部变体，先逐 serialization softmax，再按 graph_id 平均概率并 argmax。
回归验证使用所选预测；最终测试平均全部变体预测，再计算原尺度 MAE。

### MLM 验证与最佳预训练权重

MLM masking 按固定官方 `MLMDataset`：仅保护 CLS／SEP／PAD 和 attention-masked 位置，
其他特殊 ID 也可被选为目标。选中位置以 80% 概率替换为 MASK，剩余位置以 50% 概率
从完整模型词表均匀替换（包含特殊 ID），其余保持原值。两种后端使用相同概率规则；
不保证跨框架随机数轨迹逐 seed 相同。

每轮 MLM 验证对各有效 batch 的 masked-token 平均 loss 再求均值，沿用官方选优口径。
无有效目标的 batch 不参与均值；整轮无目标则记录 `loss: null`，跳过选优和早停计数。
如果整个预训练阶段没有有效验证目标，则明确失败，不读取既有 checkpoint 或进入微调。
非有限 loss／logits 立即失败，不沿用官方跳过 NaN／Inf batch 的行为。

六组 preset 的 MLM 早停耐心值均为 20，使用
`--pretrain-early-stopping-patience` 独立覆盖；表中的耐心值仍指微调。
`--pretrain-epochs` 是最大轮数。`best_pretrain.pt`（TensorFlow 为 `.npz`）保存验证最优
权重，微调前恢复该权重。summary 的 `mlm` 包含实际轮数、最佳轮次、最佳验证 loss
以及逐轮训练／验证记录，`checkpoints.pretrain` 指向配套文件。


所有 preset 均使用 `max_length=8096`、BPE `2000` merges / 最小频率 `2`、100 次序列化、
MLM mask 概率 `0.09`、warm-up `0.12 / 0.025`、梯度裁剪 `2.0 / 0.5`、默认 seeds `42–46`，
并使用 `gradient_accumulation_steps=1`、`fit_scope=all`。词表大小由拟合语料在运行时派生。

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

### 微调一致性损失（R-Drop）

按固定官方配置，Mutagenicity/GTE 和 DBLP/GTE 默认启用 R-Drop，权重为 `1.0`；
其余四组默认关闭。启用后，同一输入做两次独立的 dropout／噪声前向，分类训练使用
两次 CrossEntropy 的均值，加上 `alpha * (KL(p1 || p2) + KL(p2 || p1)) / 2`。
两个概率分支均保留梯度；每个样本先对类别求和，再按实际样本数归一化。
回归显式启用时使用两次 MSE 的均值，加上预测间 MSE 的加权项。
梯度累积和尾窗口按实际目标数归一化，不因两次前向把目标数翻倍。

CLI 使用 `--finetune-use-consistency-regularization` 启用，
`--no-finetune-use-consistency-regularization` 关闭，
`--finetune-consistency-alpha VALUE` 覆盖权重（必须有限且非负）。
微调结果记录 `use_consistency_regularization` 和 `consistency_alpha`；
验证使用每图所选变体，测试平均全部变体的概率；两者均不计算 R-Drop 损失。
任一次前向、任务损失、一致性损失、总损失或梯度出现 NaN／Inf 都终止该次实验，
不写入成功结果。离线官方 loss／梯度对照见[官方离线参考与再生成](#官方离线参考与再生成)。

### 序列长度与位置容量

`max_length` 是包含 special token 的最终 GraphTokenizer 输入上限；模型另有
`max_position_embeddings` 容量。BERT 的两个值均为 8096，配置见固定 revision 的
[`config/default_config.yml`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/config/default_config.yml#L77-L83)。
collate 会 pad 到当前 batch 中最长的序列。

GTE 的最终输入上限为 8096、位置容量为 8192；后者来自固定源码的
[`gte_model/config.json`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/gte_model/config.json#L32)。

### GTE checkpoint 与 BPE

正式 GTE 初始化由 preset 决定：AQSOL/GTE 对应官方 `encoder.reset_weights=true`，通过
`GraphGTE.from_official_config()` 加载固定架构配置并随机初始化，不读取 checkpoint；
Mutagenicity/GTE 和 DBLP/GTE 对应 `reset_weights=false`，通过 HF-to-TLX converter 加载
`Alibaba-NLP/gte-multilingual-base` revision
`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`。它验证 checkpoint SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`、官方架构、136/136
parameter coverage（包含词嵌入）和 tensor shape；任一校验失败都会停止运行。
resize 保留已有词嵌入行，扩展行使用 `N(0, 0.02)`；随机构造的线性／嵌入权重也使用该
正态分布，MLM 含独立零 bias。PAD 行清零且查表梯度为零。

现有 GammaGL 词表编号保持不变。tokenizer 在训练时额外重建官方按词频稳定排序的编号，
形成完整双射；GTE 在官方编号下初始化／resize，然后同时重排词嵌入和 MLM weight／bias。
该映射进入 tokenizer 指纹，正式路径拒绝没有官方映射的 tokenizer。

summary 将两种正式初始化分别记录为 `official_random` / `official_pretrained`，
并记录 `initialization_matches_preset: true`。初始化匹配或权重覆盖率不等于实验复现，
`reproduction` 均为 false。`--allow-random-gte-init` 保留给开发用 tiny／自定义架构，
记录为 `random_development`、`initialization_matches_preset: false`；AQSOL 的正式随机
初始化不需要该开关。初始化及训练行为的离线对照见[官方离线参考与再生成](#官方离线参考与再生成)。

`--bpe-backend python` 是默认值；`auto` 在可用时使用可选的 `graph_bpe_cpp` 原生扩展，否则
回退 Python；`cpp` 则要求该扩展并在缺失时 fail fast。无论 backend，频率、BPE 和词表
都按 `fit_scope` 选择语料；默认 all 使用全部 split 图，train 仅用训练图。
回归标签统计始终仅使用训练标签。

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

Mutagenicity 和 DBLP_v1 使用两个 logits 的 CrossEntropy；评估时每次 serialization
先计算 softmax 概率，再按图平均概率并取 `argmax` 计算 Accuracy。AQSOL 使用 MSE 训练；
其标签归一化只用 train split
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
summary 保存 seed、指标、阶段 history、checkpoint 路径及 GTE 初始化信息，并通过哈希
绑定 `run_manifest.json` 和 checkpoint。完整 CLI、生效配置、代码及数据来源记录在 manifest
及配套 preprocessing bundle 中，详见下文“原生运行产物与独立恢复”。这些自动记录只覆盖
新版 trainer 的运行，不补证此前成绩；完整日志与 dirty diff 仍需另行归档。

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

设备参数接受 `cpu`、`cuda` 和 `cuda:N`。TensorFlow 在设置随机种子和读取数据前
配置设备；直接 CLI 调用还会在导入 TLX 前配置。`cpu` 隐藏全部 GPU，`cuda:N`
选择 `CUDA_VISIBLE_DEVICES` 限定范围内的第 N 张 GPU，之后它在 TensorFlow 内部
编号为 GPU 0。GPU 使用按需显存增长。不存在的 GPU、非法参数，以及运行时已经
初始化而无法执行的设备切换都会报错退出，不静默回退。相同配置可以重复使用；
程序化调用者必须在创建 TensorFlow 张量前选择设备。

### 原生运行产物与独立恢复

每个 seed 在训练前保存 `run_XX_seed_YY/run_manifest.json`，包含实际生效配置、CLI、
seed、请求／解析后的设备、依赖版本、实际模型构造参数及初始化来源。代码来源记录
Git commit／dirty 状态与相关源码 SHA-256，不自动归档完整 dirty diff；无 `.git` 的源码快照
记录源码摘要，不推测 commit。数据清单记录 raw／实际 processed 文件 SHA-256、划分图数与编码数量；
smoke 则明确记录 synthetic 输入，不能作为真实 benchmark 的运行记录。

同组共享的 `preprocessing/<fingerprint>/` 包含 `tokenizer.pkl`（含 BPE、词表、
序列化频率与官方词表行映射）、`normalizer.json`、`data_manifest.json` 和
`bundle.json`。normalizer 继续只用训练标签，分类任务明确保存 null。
目录指纹同时绑定这些文件与模型构造配置；checkpoint 除原 tokenizer schema／
fingerprint／fit_scope 外，还保存独立 `artifact_version=1` 与产物指纹。
这不改变 serializer protocol 5、tokenizer schema 7 或 cache 9。
新运行在 manifest、seed／组 summary 中记录 `training_protocol.version=2` 及验证采样、
测试聚合、MLM masking／选优规则。checkpoint 记录相同版本；恢复前检查版本匹配。
缺少协议记录的历史运行不能与新运行混合汇总，输入 tokenizer／缓存格式不因此升级。
模型协议兼容的完整旧导出仍可按其原训练协议恢复推理，不据此重新标为新协议实验；
旧 BERT 模型需使用原代码，当前恢复入口明确拒绝不兼容的模型协议。
summary 记录 run manifest、最佳预训练及最佳微调 checkpoint 的哈希。配套文件写入或校验失败时，
本次运行不会写成功 seed summary，组 summary 保持 incomplete。

复制整个 `<dataset>/<encoder>/` 目录后，在同一 TLX 后端的新进程中恢复：

```python
from examples.graph_tokenizer.artifacts import load_run

run = load_run("copied/aqsol/bert/run_01_seed_42", device="cpu")
model = run["model"]             # 已恢复最佳权重，处于 eval 模式
tokenizer = run["tokenizer"]     # 直接加载拟合状态，不重新 fit
normalizer = run["normalizer"]   # AQSOL 用于还原原尺度；分类为 None
```

恢复不需要原始数据、预处理缓存或 Hugging Face 下载；GTE 直接按保存的实际架构
构造并加载完整下游权重，不重新应用来源编码器初始化。文件哈希、版本、模型配置、
tokenizer、normalizer 或 backend 不匹配时拒绝加载。pickle 和 torch checkpoint
仅用于可信本地产物，哈希不提供来源认证。旧 checkpoint 没有配套导出清单时，
`load_run` 明确拒绝；原低层 helper 仍可在提供原配套 tokenizer、schema／指纹兼容的前提下恢复旧格式，
不会据此宣称产物完整，也不会自动重新拟合或补默认 normalizer。

### 真实数据、正式 GTE 权重与 GPU 集成验证

以下检查不报告 benchmark 成绩。默认 CI 不下载数据或模型，也不申请 GPU；本地验证显式
提供资源后运行，配置了不存在的路径会报错，不静默跳过。

在安装 paper extra 的独立环境中，将 `GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE` 指向官方
release archive（自动校验固定 SHA-256），或已准备的可信 release 目录，然后运行：

```bash
python -m pytest tests/datasets/test_graph_tokenizer_datasets.py \
  -k real_release_bundle -q -rs
```

该测试检查三个数据集的真实加载、官方划分、节点／边标签和序列化。正式 GTE 检查另需
`GAMMAGL_GRAPH_TOKENIZER_INTEGRATION_DATA_ROOT`（包含已准备的 `mutagenicity/raw/`）和
`GAMMAGL_GRAPH_TOKENIZER_GTE_CACHE`（Hugging Face 缓存目录）。缓存中必须已有固定 revision
`9bbca17d9273fd0d03d5725c7a4b0f6b45142062` 的 config 和 safetensors，测试不联网下载：

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/models/test_graph_tokenizer_release_integration.py \
  -q -rs -o junit_family=legacy --junitxml=gte-integration.xml
```

该检查保留真实 split 成员关系，按边数选择每个 split 最小的一张 Mutagenicity 图；使用
train-only tokenizer 和完整正式 GTE 架构／预训练权重，验证 100% 编码器映射、MLM、微调、
配套产物保存及新进程离线恢复输出一致。清单明确记录 `validation_only`、小样本索引和配置，
不能将这次运行标为正式复现。两个后端读取原始 DGL release 时都需要可选 DGL 环境；
新进程恢复不需要 DGL 或模型下载。

GPU 检查只能在预留两张空闲 GPU 后显式启用，将它们的 UUID 以逗号分隔写入
`GAMMAGL_GRAPH_TOKENIZER_TEST_GPU_UUIDS`。分别在 CUDA PyTorch 和 GPU TensorFlow 独立环境中运行：

```bash
python -m pytest tests/models/test_graph_tokenizer_gpu_integration.py -q -rs
```

测试在新进程中设置 `CUDA_VISIBLE_DEVICES`，验证 CPU、`cuda:1`、实际张量／权重位置、
TensorFlow 逻辑 GPU 0 映射、重复配置、非法 GPU、初始化后切换错误，以及 tiny 训练和恢复。
PyTorch 显存上限为单卡的 10%，TensorFlow 启用按需增长，每个子进程限时 300 秒。
这些检查不会寻找、停止或修改其他实验进程。CPU CI 中它们会明确 skip，不代表 GPU 已验证。

[CI workflow](../../.github/workflows/test_push.yml) 按 `suite={core,graph-tokenizer}` 和
`backend={torch,tensorflow}` 展开四个独立单元，关闭 fail-fast。core 运行 public API、
data、utils 和 compileall；GraphTokenizer 单元额外安装转换测试依赖，运行真实后端张量、
训练／恢复、官方离线 fixtures、R-Drop、GTE 初始化和拟合范围回归，不下载模型或数据。
公共检查不依赖这些转换测试包，也不安装 DGL 或 paper extra。

矩阵使用独立的 CPU 构建基线 PyTorch 2.5.1／torchvision 0.20.1；TensorFlow 单元运行
TensorFlow CPU 2.15.1。现有 setup.py 仍需 torch 编译扩展，因此 TensorFlow 单元在安装
GammaGL 后卸载 torch／torchvision，并在测试前断言所选 TLX 后端及 torch／DGL 缺席。
torch 单元也检查 DGL 缺席。安装时的 torch 构建依赖不表示 TensorFlow 运行依赖。

独立 `graph-tokenizer-paper` job 保留 PyTorch 2.1.2／DGL 2.1.0／torchdata 0.7.1 的固定组合，
安装 paper extra、运行 DGL／GraphBolt smoke 和完整 GraphTokenizer 回归（包括拟合范围测试）。
它与矩阵单元没有 `needs` 依赖。

## 官方离线参考与再生成

六份 JSON 均由对应生成器执行固定 revision 的官方代码生成，不导入 GammaGL；
记录官方源码 SHA-256、配置、输入和参考输出。普通 CI 直接读取这些本地文件。

| Fixture | 对照范围 | 生成器 |
| :-- | :-- | :-- |
| [graph_tokenizer_official.json](../../tests/models/fixtures/graph_tokenizer_official.json) | 六组 serializer、实际 loader getter、C++ BPE、完整词表映射及增强后的模型输入 | [generate_graph_tokenizer_official.py](../../tests/models/fixtures/generate_graph_tokenizer_official.py) |
| [graph_tokenizer_fit_scope_official.json](../../tests/models/fixtures/graph_tokenizer_fit_scope_official.json) | 官方预处理入口的语料顺序、统计、BPE、词表和最终输入 | [generate_graph_tokenizer_fit_scope_official.py](../../tests/models/fixtures/generate_graph_tokenizer_fit_scope_official.py) |
| [graph_tokenizer_training_official.json](../../tests/models/fixtures/graph_tokenizer_training_official.json) | 六组验证采样配置、MLM patience、逐图采样与受控 masking 输出 | [generate_graph_tokenizer_training_official.py](../../tests/models/fixtures/generate_graph_tokenizer_training_official.py) |
| [graph_tokenizer_rdrop_official.json](../../tests/models/fixtures/graph_tokenizer_rdrop_official.json) | 官方 TaskHandler 的逐样本损失和公共标量权重梯度 | [generate_graph_tokenizer_rdrop_official.py](../../tests/models/fixtures/generate_graph_tokenizer_rdrop_official.py) |
| [graph_gte_initialization_official.json](../../tests/models/fixtures/graph_gte_initialization_official.json) | GTE resize、词嵌入、隐藏态、MLM 输出及嵌入／QKV 梯度 | [generate_graph_gte_initialization_official.py](../../tests/models/fixtures/generate_graph_gte_initialization_official.py) |
| [graph_bert_official.json](../../tests/models/fixtures/graph_bert_official.json) | BERT 完整权重、embedding／hidden／MLM／task 输出及有效参数梯度 | [generate_graph_bert_official.py](../../tests/models/fixtures/generate_graph_bert_official.py) |

GTE 参考仅将 Hugging Face 配置／checkpoint 获取替换为本地 tiny provider；
GTEEncoder、NewModel、resize 和任务头仍执行官方代码。BERT 与 GTE 初始化参考使用
Transformers 4.40.2，并记录相关实现摘要。训练参考仅控制随机抽样值和 no-BPE provider，
保留官方采样与 masking 逻辑。拟合范围参考覆盖交错 split 索引与 held-out 图影响。

### 再生成参考数据

在安装官方项目依赖的独立环境中，准备上述固定 revision 的干净源码，设置
`OFFICIAL_ROOT` 为其路径，并从 GammaGL 仓库根目录运行：

```bash
TG_THREADS_FORCE=0 python tests/models/fixtures/generate_graph_tokenizer_official.py "$OFFICIAL_ROOT"
TG_THREADS_FORCE=0 python tests/models/fixtures/generate_graph_tokenizer_fit_scope_official.py "$OFFICIAL_ROOT"
python tests/models/fixtures/generate_graph_tokenizer_training_official.py "$OFFICIAL_ROOT"
python tests/models/fixtures/generate_graph_tokenizer_rdrop_official.py "$OFFICIAL_ROOT"
python tests/models/fixtures/generate_graph_gte_initialization_official.py "$OFFICIAL_ROOT"
python tests/models/fixtures/generate_graph_bert_official.py "$OFFICIAL_ROOT"
```

生成器检查官方 revision 与相关源码状态；序列化／拟合参考会在临时目录编译官方 BPE。
模型参考使用 tiny 配置，不下载大型 checkpoint。对应回归同时覆盖 torch 与 TensorFlow
后端；测试命令和可选真实数据／GPU 集成入口见上文。

离线输出／梯度参考验证给定输入与权重下的行为，不能保证跨框架随机初始化、dropout
或完整训练的逐 seed 轨迹一致。功能回归、真实数据集成与 benchmark 成绩分别记录。

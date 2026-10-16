# GraphTokenizer 六组实验记录审计（2026-10-16）

本阶段按提供的实验确认，保留 README 六组指标作为最新版代码的复现实验结果。审计核实了
数值和现存产物，也单独记录产物版本冲突；没有启动完整训练，没有修改原始日志或成绩。

## 代码历史与当前实现

| 提交 | UTC 时间 | 实际修改 |
| :-- | :-- | :-- |
| `21d93d727eaf46cdf5b66cc39934758dba0c5a7e` | author 2026-10-08 01:22:29；commit 01:22:54 | 改动 13 个文件，引入 serializer protocol／tokenizer schema 3、遍历局部引用及 `local_to_original`；按唯一训练图统计 AQSOL 标签；升级缓存、校验下游 checkpoint；增加测试。README 当时将表中数值标为旧协议，并写明修复后待重跑。 |
| `c4e2810fe55f7d32da72f4ff904beb16decad2fb` | author／commit 2026-10-08 02:19:13 | 是前一提交的直接子提交，仅改 README。保留六组数值，改为当前 benchmark 表，删除旧结果与待重跑说明，同时删除必要的兼容性说明。没有改训练实现，也没有纳入原始实验附件。 |
| `18f6ebf6c94d9738ea0b51b1ccd42d16ae4e2722` | author 2026-10-16 05:19:44；commit 05:20:31 | 本次文档审计的代码基线。无节点特征时使用全零标签；预处理缓存跟踪实际 processed 数据与 split 的 SHA-256；serializer protocol 4、tokenizer schema 4、缓存格式 5。 |

早期“仍需重跑”的文档状态不能覆盖本次对最新版六组实验已完成的确认。与此同时，
`c4e2810` 的文档改动本身也不是重跑的原始记录。现有材料尚不能把最新版运行精确绑定到
某个 commit；本次审计基线 `18f6ebf` 不能填作未知的实验执行 commit。

当前源码已核对：

- 节点引用按最终组件顺序及遍历首次访问顺序连续编号，原始编号保存在 provenance metadata。
- `x=None` 使用零标签，恢复出的 `x` 仍是默认标签；显式整数标签保持原值。benchmark
  Dataset 要求提供节点特征，此默认行为属于通用 serializer 的接口。
- AQSOL 只在 train split 按唯一 `graph_id` 统计一次标签，检查同图不同 serialization 标签
  一致性，使用训练标签总体标准差归一化。评估先平均同图各次预测，再恢复原尺度计算 MAE。
- 分类先平均同图各次 logits，再取 `argmax` 计算 Accuracy；验证集选最佳 checkpoint，恢复
  后仅进行一次最终测试。跨五 seed 的汇总使用 `statistics.stdev`，即 `ddof=1`。
- 不支持任意图的全局同构规范化；结构签名和组件排序打平时仍可能受原始编号影响。

## 六组记录的实际位置

以下路径位于审计环境，未随仓库发布。A、B 的目录名只用于定位，不作为代码版本证据：

```text
A = /local/wrq/graph-tokenizer-pr263-formal-20260922
B = /local/wrq/graph-tokenizer-pr263-formal-20260923-retry
数据根目录 = /local/wrq/graph-tokenizer-pr263-data-20260922
```

| 实验 | 数值匹配的完整记录目录 | 日志 | 可核实的 batch／累积覆盖 |
| :-- | :-- | :-- | :-- |
| Mutagenicity + BERT | `A/mutagenicity/bert/` | `A/logs/mutagenicity_bert.log` | scheduler 未记录具体参数，不能把当前默认值当作当时 CLI |
| Mutagenicity + GTE | `B/mutagenicity/gte/` | `B/logs/mutagenicity_gte.log` | 预训练／微调 batch `2 / 2`，梯度累积 `16` |
| DBLP_v1 + BERT | `A/dblp/bert/` | `A/logs/dblp_bert.log` | scheduler 未记录具体参数，不能把当前默认值当作当时 CLI |
| DBLP_v1 + GTE | `B/dblp/gte/` | `B/logs/dblp_gte.log` | 预训练／微调 batch `2 / 2`，梯度累积 `16` |
| AQSOL + BERT | `B/aqsol/bert/` | `B/logs/aqsol_bert.log` | 预训练／微调 batch `8 / 8`，梯度累积 `4` |
| AQSOL + GTE | `B/aqsol/gte/` | `B/logs/aqsol_gte.log` | 预训练／微调 batch `2 / 2`，梯度累积 `16` |

每个目录均有 `summary.json` 和 `run_01_seed_42/` 至 `run_05_seed_46/summary.json`，并有
各 seed 的 `best_finetune.pt`。本次检查了全部 30 份 per-seed summary、六份汇总、六份训练
日志完成行和对应 scheduler 日志；逐 seed 的 `metric`、`finetune.test.metric`、日志输出及
汇总 `run_metrics` 相符。不能仅用 `complete: true` 判断归属，因此另外检查了 history、
版本字段和缓存内容。

六组均记录 MLM 完成 200 epochs；微调 history、最佳 epoch、验证成绩和测试图数均保存在
per-seed summary。三组 GTE 的全部 15 份初始化 manifest 记录固定 revision
`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`、checkpoint SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`、135/135 encoder
tensors 和 `reproduction: true`；这些是初始化记录，不是 GraphTokenizer 输入协议凭据。

本地 raw split 的 train／val／test 图数分别为 Mutagenicity `3469 / 434 / 434`、DBLP_v1
`15564 / 1946 / 1946`、AQSOL `7858 / 983 / 982`。per-seed summary 的测试图数分别为
434、1946、982，与之相符。审计时的 split 文件摘要已记录，但运行时未保存这些摘要，
因此不能仅据当前文件位置确认运行期间的数据内容没有变化。

四个 retry 实验的 scheduler 行与 `B/run_dynamic_gpu_job.sh` 可核对 batch／累积覆盖。
这使它们与当前 README **默认 preset** 的区别可见；完整生效配置没有随 summary 归档。
当前 preset、默认模型维度与源码内容摘要另存于审计 JSON，只描述当前源码，不能回填为
这些运行已经核实的配置。学习率、mask／augmentation、warm-up、最大微调轮数等参数的
实际覆盖仍需运行时配置快照确认。

## 逐 seed 指标和统计复算

下表 Accuracy 全部采用百分数，标准差单位为百分点。为便于核对，审计表显示八位小数；
README 的 Accuracy 显示两位、MAE 显示四位。[审计 JSON](experiment_records/audit_2026-10-16.json)
保留 summary 中原精度的指标（Accuracy 原始值仍为 0–1），并记录每份来源文件的 SHA-256。

| 实验／指标单位 | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | mean | sample std |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Mutagenicity + BERT（Accuracy %／std 百分点） | 76.72811151 | 79.03226018 | 77.64977217 | 75.80645084 | 74.42396283 | 76.72811151 | 1.75478766 |
| Mutagenicity + GTE（Accuracy %／std 百分点） | 57.60368705 | 70.50691247 | 70.96773982 | 70.96773982 | 70.50691247 | 68.11059833 | 5.87805968 |
| DBLP_v1 + BERT（Accuracy %／std 百分点） | 91.00719690 | 93.88489127 | 93.21685433 | 92.08633304 | 91.05858207 | 92.25077152 | 1.28437698 |
| DBLP_v1 + GTE（Accuracy %／std 百分点） | 91.52106643 | 92.70297885 | 92.08633304 | 92.39465594 | 91.98355675 | 92.13771820 | 0.44502871 |
| AQSOL + BERT（MAE） | 0.74903524 | 0.73692471 | 0.73072821 | 0.73645681 | 0.74434489 | 0.73949797 | 0.00719776 |
| AQSOL + GTE（MAE） | 0.71281242 | 0.71494222 | 0.70862311 | 0.69778740 | 0.70417380 | 0.70766779 | 0.00689433 |

复算方法是先从五份独立 per-seed summary 读取最终测试指标，计算 mean 与
`statistics.stdev`，再对 Accuracy 的均值和标准差同时乘以 100；不对已舍入的 README
显示值重新统计。所有结果均与原 README 数值的显示精度一致。Mutagenicity + GTE 的
sample std 复算为 `0.05878059684023406`，汇总 JSON 为 `0.05878059684023405`，差异仅在
浮点末位，不影响 `5.88` 个百分点的显示。其他汇总值在本次 Python 复算中一致。

论文／官方参考栏不参与上述复算；原始汇总、seed 成绩和日志均未改写。

## 本地材料的版本冲突与证据边界

六组日志都给出了实际使用的 preprocessing cache 路径；本次只读检查这些路径下的文件，
发现六个 tokenizer 均为 **schema 2**，payload 没有 `preprocessing_cache_version`。
它们的实际路径、审计时 SHA-256、BPE 配置及版本观测保存在 JSON。缓存中的 BPE metadata
记录 Python backend、2000 merges、最小频率 2，tokenizer 记录 100 次 realizations。

这不仅是早期 commit 中的一句话：Mutagenicity 缓存首个训练图的首次访问引用从
`0, 1, 4, 5, 2, ...` 开始，AQSOL 从 `0, 11, 12, 13, 1, ...` 开始，不符合当前局部引用
首次访问必须连续编号的规则。两组 DBLP 缓存的首图引用恰好连续，不能凭这一单图推断协议；
其 tokenizer 的持久化 schema 仍为 2。

两组 AQSOL 缓存保存的 normalizer 为 mean `-3.398347088909526`、std
`2.4255524263154937`。从缓存 labels 按其 normalizer 还原并按唯一训练 `graph_id` 统计，
得到 mean 约 `-2.8736468650446922`、总体 std 约 `2.361552813625175`，与缓存值不同。
因此这些现存缓存不符合当前按唯一训练图统计的实现。

全部 30 个本地下游 checkpoint 只有 `model`、`epoch`、`score` 字段，没有
`tokenizer_schema_version`；本次使用 `torch.load(map_location="meta", weights_only=True,
mmap=True)` 只读检查 metadata，未进行模型训练或推理。它们会被当前恢复函数的版本检查拒绝。

这些内容是**数值匹配的本地实验材料存在旧格式／旧输入特征的直接证据**，应与最新版结果的
确认同时保留。summary 未记录训练代码 commit 或 tokenizer 协议，日志也没有当时缓存文件
的内容摘要；本次 SHA-256 只能固定审计时看到的文件。尚不能把这批材料证明为最新版重跑的
产物，也不能排除另有尚未提供的最新记录。README 保留当前六组成绩及其确认的归属，同时
明确其最新代码／协议关联尚未独立核实，不用这批材料替代最新运行凭据。

## 兼容性核对

当前代码没有自动迁移旧产物：

- tokenizer 的 `_require_fitted()` 检查 schema 4 和训练原始 token 集；完整 result 解码检查
  tokenizer schema／serializer protocol 4。裸整数流没有版本 metadata，无法自动识别来源协议。
- standalone `GraphBPE.load_codebook()` 只加载 merges、vocab size 和 metadata，不校验输入协议，
  也不等同于完整 tokenizer 的词表与 provenance。
- 预处理缓存键包含实际数据和 split 内容、协议与预处理配置；payload 校验格式 5 和 schema 4。
  旧键不会命中，当前键下的不兼容 payload 不复用；写入保留原子替换。它不自动归档实验 commit。
- Dataset 当前读取 `torch_data_v2.pt`／`split_indices_v2.json`；现存旧数据目录中的
  `torch_data.pt`／`split_indices.json` 不是这些新文件的替代，缺少当前文件时需由 raw 重新处理。
- 下游 checkpoint 在 `model.load_state_dict()` 前检查 schema 4；缺失或不一致即报错。
  检查没有覆盖具体 tokenizer 词表和数据指纹，同 schema 仍需配套产物，不能只比较形状。
- GTE 来源 checkpoint 的固定 revision、SHA-256、配置、tensor shape 和 135/135 映射检查
  独立执行；这不是对旧 GraphTokenizer 下游权重的兼容或迁移。

## 需要补齐的最新版运行记录

六组分别尚缺能绑定最新版重跑的：完整 Git commit／源码快照及 dirty diff、完整 CLI 与生效
配置、运行时 raw／processed／split 内容摘要、tokenizer／BPE／缓存与下游 checkpoint 指纹。
新提供的记录应逐 seed 对应训练、最佳验证 checkpoint 和最终测试，再与当前表格核对。
当前 trainer 的 summary 不自动记录这些全部字段；本文列出的是归档要求，没有新增自动
采集或迁移逻辑，也没有推断未知版本号。

## 本阶段验证

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

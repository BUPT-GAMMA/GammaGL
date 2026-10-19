# PR #263：固定官方协议核对

依据为 [官方 revision 98343a6b025a48fbb6859cd812a12b81ec3ac3cc](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)。修复前基线为 `78b92f5`。
本阶段只验证协议与功能；不重跑完整 benchmark，不修改六组已报告数值或结果归属，
也不将这些记录视为本次代码的验证结果。

实际调用链已检查：`ProjectConfig` 加载 `default_config.yml`，`config_override.py`
递归覆盖 JSON；`UnifiedDataInterface._build_and_persist_serialization()` 创建 serializer，
调用 loader 的批量／单节点 getter；`prepare_data_new.py` 训练 C++ BPE 并构建词表；
`ClassificationDataset.__getitem__()` 先增强原始 token，再 BPE、词表编码；
`TaskHandler.get_probabilities()` softmax，`evaluate.py` 按 graph_id 聚合。
fixtures 记录了这些源码文件与配置文件的 SHA-256。

| 项目 | 官方行为 | 修复前 | 修复后 |
| :-- | :-- | :-- | :-- |
| 六组 serializer | DBLP GTE JSON 为 eulerian；Mutagenicity GTE JSON 为 feuler；其余四组继承默认 feuler | make_tokenizer 统一 Feuler | 六组显式选择，CLI 传入实际构造及缓存键 |
| 实际 BPE 编码 | 两个 TU GTE JSON 为 none；其余继承 all；预处理仍训练规则 | 始终压缩 | 显式 all／none，训练规则与模型编码分别执行 |
| Eulerian | 保留输入弧；按本地节点 ID 升序排序后 pop 最大；仅总入出度存在奇数时复制弧 | 无条件规范化成双向弧，按边标签／结构签名排序 | 按官方条件和遍历顺序执行 |
| Feuler | 有向 triplet 统计转 float32 log10 权重，频率降序／邻居编号升序 | 无向模式计数，结构签名打平 | 定向统计及官方编号排序，不注入结构签名 |
| 节点 token | TU 批量 getter 为 2*node_token_ids+1，singleton getter 为原值；AQSOL 为 2*atomic_num+1 | 再包装标签并添加局部引用 | 实现这些实际 getter 的规则，无额外引用 |
| 边 token | TU 批量 getter 为 2*edge_token_ids，省略类型 0；AQSOL 为 2*bond_type，省略 SINGLE=1 | 包装所有边标签 | 仅排列官方保留的边 token |
| 组件与起点 | 组件按节点数量降序；非连通图每个子图起点重置为本地 0；多起点按原节点编号均匀取样 | 按遍历长度及标签签名排序；按首次访问顺序取样 | 官方顺序、起点与普通 token 7 组件分隔 |
| BPE 学习对象 | 原始完整序列；首个 merge ID 为 max(base)+2；词表另编码 | 引用流、偏移后的标签、受保护低编号及 max(base)+1 | 原始流、无低编号保护、相同 merge 规则和编号 |
| 词表特殊边界 | 官方 VocabManager 永久预留 -pad_id=0，所以普通原始 0 也映射 PAD | 标签被包装，绕过该实际行为 | 保留该 alias，即使训练语料未出现原始 0 |
| 训练增强 | 对原始序列所有位置做窗口内 swap／mask，再编码 | 把模型特殊 ID 集用于原始序列，错误保护普通 token；swap 抽样也不同 | raw 无该保护；窗口和 mask 抽样按实际官方实现，fixtures 比较 loader 最终输入 |
| 分类 avg | 每个 serialization softmax，按 graph_id 平均概率，再 argmax | 平均 logits | 验证选优、最终测试共用概率平均 |
| 回归 | 平均标准化预测后还原原尺度并计算 MAE | 已正确 | 保持，覆盖不等变体数量及 batch 边界 |
| 图恢复 | 官方 base deserialize 抛 NotImplementedError；标签流没有节点身份 | 声称默认流可完整恢复 | official 的三个恢复接口明确拒绝；可逆引用仅为显式 local_refs 协议 |
| 拟合语料 | BaseDataLoader.load_data 拼接 train + val + test；UDI 对全部图拟合统计并生成 BPE 训练序列 | 仅 train | 六组 preset 默认 all，保持上述顺序；train-only 为独立变体；标签归一化仍只用训练标签 |
| 产物兼容 | 本地移植须绑定其输入约定 | protocol/schema 4、cache 5；checkpoint 只检查 schema | protocol 5、schema 7、cache 9；缓存配置、实际 fit_scope 和指纹复检；checkpoint 在载权重前检查范围及具体 tokenizer 指纹 |

## 离线参考输出

[`graph_tokenizer_official.json`](../../tests/models/fixtures/graph_tokenizer_official.json)
由 [独立生成脚本](../../tests/models/fixtures/generate_graph_tokenizer_official.py) 执行上述固定
revision 的**未修改** serializer、实际数据 loader getter、C++ minBPE、BPEEngine、
VocabManager 和 transforms 生成。生成器不导入 GammaGL，先检查 commit 和源码工作区，
编译官方 BPE 到临时目录；不加载大型 checkpoint。手工再生成命令：

```bash
TG_THREADS_FORCE=0 python tests/models/fixtures/generate_graph_tokenizer_official.py /path/to/pinned/official
```

六组各用相同的六张显式小图：分叉、带标签环、大小不等组件及孤立节点、singleton、
频率引导路径、单向 COO 环。图的节点／弧／标签、配置来源、起点、8 次 merge／最小频率 2、
完整原始序列、merge rules、压缩序列和官方最终 input_ids 都存入 JSON；每图另有三个
固定 seed 的 swap／mask 输出及最终输入。
单节点和组件中 singleton 的不同 getter 路径都实际执行。

原始 token 与 BPE 编号逐项相等。模型词表编号采用固定双射：GammaGL 的普通 token 按
原始／merge ID 升序分配，官方按词频分配；每组 JSON 的
`official_to_gammagl_model_id` 显式列出**整个词表**的对应关系，覆盖所有 special IDs，
GTE 的 PAD/UNK 交换以及普通 0/PAD alias。测试验证其定义域和值域均为完整连续词表，
逐项比较完整模型输入，未删除或过滤任何 token。该映射绑定指定训练语料，不能跨词表复用。
普通 CI 仅读取本地 JSON，覆盖 Python 和可选 native BPE，无在线 clone 或模型下载。

## 本次验证

环境：Python 3.10，PyTorch 2.1.2 CPU／TLX torch，以及独立 TensorFlow eager／TLX 环境。
本地编译可选 GammaGL C++ BPE 扩展，两个 backend 均执行全部官方 fixtures。

```bash
TL_BACKEND=torch python -m pytest -q -rs \
  tests/transforms/test_graph_tokenizer.py tests/transforms/test_graph_tokenizer_backend.py \
  tests/datasets/test_graph_tokenizer_datasets.py tests/models/test_graph_tokenizer_models.py \
  tests/models/test_graph_gte_backend_conversion.py tests/models/test_graph_tokenizer_dataset_metadata.py \
  tests/models/test_graph_tokenizer_tlx_training.py tests/models/test_graph_tokenizer_official_protocol.py \
  tests/models/test_graph_tokenizer_pipeline.py

TL_BACKEND=tensorflow python -m pytest -q -rs \
  tests/models/test_graph_tokenizer_tlx_training.py tests/transforms/test_graph_tokenizer.py \
  tests/transforms/test_graph_tokenizer_backend.py tests/models/test_graph_gte_backend_conversion.py
```

实际结果：torch **382 passed, 1 skipped**；TensorFlow **115 passed**。唯一跳过项是未配置
真实 release bundle 的测试。数值异常测试有两条预期 NumPy warning，异常均被正确拒绝；
另有 PyTorch 的 pkg_resources 弃用 warning。梯度累积、全空 MLM 窗口、尾窗口、非有限
指标／梯度／权重及旧协议拒绝测试均保留。独立分类反例 `[[10,0],[0,2],[0,2]]` 得类别 **1**，
覆盖验证／测试、单变体、交错 graph_id、不等变体数及 batch 1／2／4／6。

另运行 torch／TensorFlow × BERT／GTE 的四次 AQSOL tiny smoke（seed 42）：MLM、
微调、验证选优、恢复和最终测试均完成，checkpoint schema=5，均绑定非空 tokenizer
指纹且 score 有限。使用 `--smoke --dataset aqsol --encoder bert|gte --seeds 42`，
GTE 显式加 `--allow-random-gte-init`；这些是开发 smoke，不是正式 GTE 权重或 benchmark 验证。

## 后续训练目标与初始化修复

- 微调 R-Drop 已按两个 TU/GTE JSON 默认启用（alpha=1）；其余四组继承关闭。
  `tests/models/fixtures/graph_tokenizer_rdrop_official.json` 由配套的
  `generate_graph_tokenizer_rdrop_official.py /path/to/pinned/official` 直接执行
  未修改的 `TaskHandler` 生成，保存六组配置及源码 SHA-256、固定两组 logits 的
  逐样本 loss 和公共标量权重梯度。torch / TensorFlow 离线测试核对两次任务损失均值、
  双向 KL（两个分支均求导）、关闭 / alpha=0 / 非默认权重、回归 MSE，以及不等 batch
  和累积尾窗口。这证明给定两组输出时目标和梯度一致，不证明跨框架 dropout 随机轨迹
  或完整训练等价；其余训练设定差异仍需独立解决。

- GTE 初始化按 preset 对齐：AQSOL/GTE 使用官方配置随机构造；两个 TU/GTE 加载预训练
  编码器并保留 resize 后的词嵌入。均使用正态初始化（非 TF 截断正态），PAD 行清零且
  查表梯度为零。新 MLM head 使用独立零 bias，与未修改的 UnifiedTaskHead 一致。
  GammaGL 编号保持原样；fit 额外重建官方词频排序，完整双射同时用于词嵌入和 MLM
  weight / bias，并进入 schema 6 指纹。cache 8 废弃缺少行映射的旧缓存。
  summary 分开记录初始化匹配；不因覆盖率 100% 就标记实验 reproduction=true。

  `graph_gte_initialization_official.json` 由配套生成器执行固定 revision 的
  GTEEncoder、NewModel、HF resize 和 UnifiedTaskHead 产生，只把配置／权重获取替换为
  本地 tiny provider，源码不修改、不导入 GammaGL。包含源码 SHA-256、源权重、缩小／
  扩展结果、真实 input_ids、嵌入／编码器／MLM 输出及嵌入／QKV 梯度。两个 TLX 后端
  离线比较实际 token 查表路径，另验证完整行置换、MLM bias 和随机构造的配置／正态尾部。
  给定权重和输入下的数值对照不承诺跨框架随机初始化／dropout 的逐 seed 轨迹一致。
  源项目未锁定 Transformers；该 resize 参考明确使用 Transformers 4.40.2，并保存版本及
  resize 实现的 SHA-256，新增行使用正态初始化。正式 TU 词表小于来源文本词表时仅保留
  既有行，不依赖不同 Transformers 版本的扩展行策略。

  本轮回归结果：torch **504 passed / 3 skipped**，TensorFlow **279 passed / 5 skipped**。
  另外在两个后端均完成正式 768 维、12 层架构的离线 CPU 初始化与前向检查：AQSOL 为
  official_random，DBLP 为 official_pretrained；后者实际校验缓存 checkpoint SHA-256、
  加载 136/136 个张量，并逐项比较配套词嵌入保留行。输出均有限。TensorFlow 环境无 torch；
  其可选 huggingface-hub 依赖单独安装。全库 pytest 收集仍被既有 llmrec/utility 导入错误
  阻塞；compileall 与 git diff --check 通过。这些检查不替代完整 benchmark。

## 拟合范围修复

六组 preset 默认 `fit_scope=all`，统一通过 `fit_tokenizer` 按 train、val、test 顺序
选取图，再拟合频率／BPE／词表。该顺序来自官方 `BaseDataLoader.load_data()` 对
`_all_data` 的赋值，会影响词频平局下的词表编号；不是原始 data.pkl 的存储顺序。
标签不参与这些拟合，回归归一化继续仅使用训练标签，MLM／微调只用训练 split。
这属于 transductive 预处理。

`--fit-scope train` 保留为独立 inductive 变体，产物写到 encoder 目录下的 `train_only/`，
summary 记录实际范围和 tokenizer 指纹，不同范围的运行不能混合汇总。通用
`GraphTokenizer.fit` 默认仍为 train；调用方负责提供对应语料。schema 7／cache 9
废弃缺少范围的旧产物；即使给定语料相同，train／all 的 tokenizer 指纹也不同。
cache payload 复检实际 tokenizer 范围，checkpoint 在赋权重前额外检查显式范围。

新增 [`graph_tokenizer_fit_scope_official.json`](../../tests/models/fixtures/graph_tokenizer_fit_scope_official.json)
由 [生成器](../../tests/models/fixtures/generate_graph_tokenizer_fit_scope_official.py) 执行未修改的
官方 `prepare_data_new._prepare_one_mp`，完整经过实际 loader、UDI、serializer、C++ BPE、
词表构建与保存入口。仅将 ProjectConfig 的创建指向临时 CPU 配置，factory spy 只保留
对象用于读取统计。无 GammaGL 导入，保存固定 revision 和源码 SHA-256。
使用交错索引 train=[2,0]、val=[3]、test=[1]，断言官方实际语料顺序为 [2,0,3,1]。
六组离线测试从 native 数据集入口加载同一输入，逐项比较序列、merge rules、完整词表
编号、映射后的最终输入及定向频率（AQSOL 类型统计按已编码 token 对齐），并验证 held-out
图改变训练图遍历、BPE 和词表；改变 held-out 标签不改变 tokenizer 或归一化。

```bash
TG_THREADS_FORCE=0 python tests/models/fixtures/generate_graph_tokenizer_fit_scope_official.py /path/to/pinned/official
TL_BACKEND=torch python -m pytest -q tests/models/test_graph_tokenizer_fit_scope.py
TL_BACKEND=tensorflow python -m pytest -q tests/models/test_graph_tokenizer_fit_scope.py
```

本轮相关回归覆盖 public API、tests/data、GraphTokenizer transform／数据集／模型／初始化、
trainer／协议和上述范围测试：torch **486 passed / 2 skipped**，TensorFlow
**261 passed / 7 skipped**。跳过项为下载外部 PPI、未配置真实 release bundle 和后端专属测试。
两种后端均完成 all／train × BERT／GTE × AQSOL／Mutagenicity 的 tiny smoke，另覆盖
all／train × 预训练／微调／验证／权重异常的失败 summary 范围记录。独立只读复核发现的
失败 summary 字段及 README 残留说明均已修复，复核无剩余发现。
compileall 和 git diff --check 通过；未运行正式 benchmark。

## CI 矩阵修复

`.github/workflows/test_push.yml` 的 `build-and-test` 以
`suite={core,graph-tokenizer}` × `backend={torch,tensorflow}` 展开四个独立环境，
`fail-fast=false`，一个单元失败不会取消另一后端。core 仅安装公共依赖并运行
compileall／public API／data／utils；GraphTokenizer 单元独立安装 safetensors、
huggingface-hub、PyYAML，执行真实后端回归及全部离线官方 fixtures，包括前述拟合范围测试。
原生 BPE 可用性有显式断言，避免 C++ 路径因扩展未构建而静默跳过。

公共构建使用独立的 PyTorch 2.5.1／torchvision 0.20.1 基线；TensorFlow 运行使用
tensorflow-cpu 2.15.1。setup.py 当前仍编译 torch extension，因此 TensorFlow 单元
安装包后卸载 torch／torchvision，并断言 torch 不存在；所有矩阵单元都断言 DGL 不存在
及 TLX 后端匹配。这里没有改变源构建需 torch 的现有安装契约。

`graph-tokenizer-paper` 独立保留 PyTorch 2.1.2／DGL 2.1.0／torchdata 0.7.1，
安装 paper extra，检查 DGL pickle／GraphBolt 运行时并执行同一组 GraphTokenizer 回归。
它不作为公共矩阵的 `needs` 前提。后端专属测试保留明确的 skip 条件，真实数据包仍需 opt-in。

本地验证使用两个全新 Python 3.10 venv 和两个独立源码副本，初始化固定子模块
`8540627c91b818df599a9bcf31cb7179f69b56bc`，实际按 workflow 执行安装／编译、卸载、
隔离检查与测试命令。首次副本未带子模块导致的缺头文件失败已纠正，两个环境均重新
完整构建成功，没有复用原工作区的二进制。公共测试先在未安装转换依赖的环境执行。
GAMMA TLX nightly 实际解析到 `8fbe30d0588a5e58673d9cfad77b111af93182bb`，
NumPy 1.26.4；转换测试使用 safetensors 0.8.0、huggingface-hub 2.2.0。

| 单元 | 本地结果 |
| :-- | :-- |
| core / torch 2.5.1 CPU | 55 passed / 2 skipped；compileall 通过 |
| core / TensorFlow CPU 2.15.1，无 torch／DGL | 54 passed / 3 skipped；compileall 通过 |
| GraphTokenizer / torch 2.5.1 CPU，无 DGL | 467 passed / 1 skipped |
| GraphTokenizer / TensorFlow CPU 2.15.1，无 torch／DGL | 243 passed / 5 skipped |
| paper / torch 2.1.2 CPU，复用已校验的固定环境 | DGL smoke 通过；467 passed / 1 skipped |

原生回归设置 `HF_HUB_OFFLINE=1` 和独立空 `HF_HOME`，结束后两处缓存文件均为 0；
转换与原生 BPE 路径没有因缺包而跳过。实际跳过项仅为可选 RDKit、外部 PPI／真实
release bundle 以及明确限定 torch 的测试。矩阵环境最终 `pip check` 均通过；
actionlint 及 git diff --check 通过。以上为本地安装和命令验证，不是远端 GitHub
Actions 的运行记录，也不替代正式 benchmark。

## 共享 Transformer 组件整理

新增 `gammagl/models/_graph_transformer.py`，以纯 TLX 函数复用参数计数、masked mean、
additive attention bias、拆头／合头和 scaled dot-product attention。BERT、GTE 均直接
依赖该内部模块，GTE 不再从 BERT 模块借用函数。共同注意力顺序保持为 QK／sqrt(D)、
mask bias、softmax、原有 dropout 层、乘 V；全 mask 行沿用原有有限 -10000 bias 语义。
模块不创建参数或子模块；投影、位置编码、FFN、任务头及各自的初始化留在原模型中。
BERT 截断正态／GTE 正态的差别保持，参数层级、公开模型导出、schema 7／cache 9 不变。
GTE 的直接文件加载 ImportError 回退此前已删除，本轮确认没有恢复；模型和组件测试
均通过正常包导入执行。

新增 `test_graph_tokenizer_components.py`，先观察缺少共享模块时三个测试失败，再实现后
通过；最后补充实际 Dropout 的 train／eval 测试。持久测试覆盖两个 batch 的拆头布局、
合头还原、空行池化、mask bias、解析注意力值、被 mask 的 V 梯度，以及 dropout 在概率
与 V 相乘前执行。矩阵原生回归和独立 paper job 均接入该文件。

重构前记录 BERT／GTE × mean／CLS 的 tiny 模型；重构后在同一后端、同一 seed 下，
逐项比较初始权重、参数名称、manifest、全部 forward 输出及可达参数梯度，共 317 项记录，
并将权重改动后恢复重构前保存的 trainer checkpoint。Torch 2.1.2、Torch 2.5.1、无 torch 的
TensorFlow 2.15.1 三个环境均逐元素完全相同，旧 checkpoint 均恢复成功。这是本次重构
兼容性检查，不声明跨后端初始化轨迹相同。该前后对照为本地临时脚本验证；持久回归还
包括既有官方 GTE 隐藏态／QKV 梯度 fixture 及模型的有限梯度、padding 和训练恢复测试。

本轮实际执行 workflow 的命令：公共测试 torch **55 passed / 2 skipped**、TensorFlow
**54 passed / 3 skipped**；原生 GraphTokenizer 回归 torch **471 passed / 1 skipped**、
TensorFlow **247 passed / 5 skipped**。两种原生环境继续检查 DGL 缺席，TensorFlow 检查
torch 缺席；Hugging Face 离线且缓存为空。paper job 的 DGL smoke 及
**471 passed / 1 skipped** 回归通过。独立只读审查未发现阻断问题。

另执行全库 `python -m pytest -q`：原工作区未构建 sparse 扩展，导致
`tests/loader/test_neighbor_sampler.py` 和 `tests/loader/test_node_neighbor_loader.py`
在收集时失败；在已完整构建的独立副本中这两处通过收集。副本仍在
`examples/llmrec/utility/batch_test.py` 收集时因 `ModuleNotFoundError: utility` 中断。
因此只报告相关回归通过，不声称全库测试通过；上述范围外问题未随本次重构修改。

## 剩余差异与验证边界

- 为显式 train 变体及语料外输入，未见原始 token 使用独立 OOV sentinel，避免冒充数值
  相同的 merge ID，最终进入 UNK；普通 0 依官方永久 alias 单独保留。
- GammaGL 词表采用上述显式双射；独立 GraphBPE 的 vocab_size 表示 ID 上界，官方引擎
  的同名统计表示词汇数量，不用它判断规则或模型输入是否相同。
- 仅支持简单无向语义和标量 token；不扩展官方实现可接受的多维 token／多重图范围。
  GammaGL 超长输入明确拒绝，增强后超长回退原输入；官方词表编码器会截断。
- 本次未验证 GPU、真实 release bundle，没有重跑六组 benchmark。
  旧成绩保持原记录，不能证明修复后的代码达到相同结果。

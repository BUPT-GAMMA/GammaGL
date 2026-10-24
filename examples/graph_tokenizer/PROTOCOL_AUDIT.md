# GraphTokenizer 官方协议与离线参考

本实现对照[固定官方 revision `98343a6b025a48fbb6859cd812a12b81ec3ac3cc`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)。
本文记录当前协议和离线参考的来源；训练命令、六组 preset、产物布局与测试成绩见
[README](README.md)。

## 当前协议

| 项目 | 当前约定与官方依据 |
| :-- | :-- |
| 序列化 | DBLP/GTE 使用 Eulerian，其余五组使用 Feuler。Eulerian 保留输入弧及官方遍历顺序；Feuler 使用定向 triplet 频率、float32 log10 权重及官方编号排序。 |
| Token | TU 批量与 singleton getter 分别按官方规则编码；AQSOL 使用原子序数和键类型。保留官方常见边类型省略、组件排序及普通 token 7 分隔规则。 |
| BPE | 在原始完整序列上学习，首个 merge ID 为 max(base)+2。两个 TU/GTE preset 拟合规则但不压缩模型输入，其余四组应用 BPE。Python 和可选 C++ 路径均由离线参考校验。 |
| 词表 | GammaGL 普通 token 按 ID 排序，官方按词频排序；用覆盖完整词表的双射对照最终模型输入，并映射 GTE 词嵌入与 MLM 权重／bias。保留普通 raw-0／PAD alias。 |
| 拟合语料 | 官方 BaseDataLoader 按 train → val → test 拼接图；六组 preset 默认 `fit_scope=all`，属于 transductive 预处理。标签归一化只用训练标签；`train` 为产物与汇总隔离的变体。 |
| 训练增强 | 在原始 token 上按官方窗口 swap／mask，再 BPE 和词表编码。原始 token 与模型特殊 ID 属于不同编号空间。 |
| 验证与测试 | MLM／下游验证每轮每图随机选一个变体；最终测试遍历全部变体。分类先 softmax 再平均概率，回归平均预测后还原标签尺度并计算 MAE。 |
| MLM | 仅保护 CLS／SEP／PAD 和 attention-masked 位置；随机替换从完整词表抽样。按有效验证 batch loss 的均值选优，patience 默认 20，微调前恢复最佳 MLM 权重。 |
| R-Drop | 两个 TU/GTE preset 默认启用，alpha=1；其余四组关闭。任务损失取两次前向均值，双向 KL 的两个分支均求导。 |
| 初始化 | AQSOL/GTE 按官方配置随机初始化，两个 TU/GTE 加载固定来源的预训练编码器。BERT 使用普通正态 N(0, 0.02) 和独立零初始化 MLM bias；PAD 词嵌入行清零且查表梯度为零。 |
| 产物兼容 | serializer protocol=5、tokenizer schema=7、cache=9、training protocol=2、BERT model protocol=2。缓存、checkpoint 与汇总检查相应协议、fit_scope 和 tokenizer 指纹。 |

梯度累积按实际目标数归一化；MLM 空窗口同时跳过 optimizer 和 scheduler。
NaN／Inf 的 loss、logits、梯度、权重或指标会终止运行。空 MLM 验证 batch 不参与选优，
整个预训练阶段均无有效验证目标则失败。成功运行绑定 tokenizer／normalizer、两阶段
checkpoint 及 SHA-256，并支持迁移目录后的离线恢复。

## 离线参考与生成器

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

### 再生成

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
后端；完整测试命令和可选真实数据／GPU 集成入口见 [README](README.md)。

## 验证边界

- 官方 token 流缺少节点身份并省略部分边类型，三个图恢复接口明确拒绝恢复。
  可逆引用仅属于显式 `local_refs` 变体。
- 支持简单无向语义和标量 token，保留输入 COO 弧。超长图输入明确拒绝；
  官方词表编码器会截断。未见原始 token 使用独立 OOV sentinel 后映射 UNK，避免冒充 merge ID。
- 词表双射绑定具体拟合语料；独立 GraphBPE 的 `vocab_size` 表示 ID 上界，
  不用它代替完整规则及模型输入对照。
- 离线输出／梯度参考验证给定输入与权重下的行为，不能保证跨框架随机初始化、
  dropout 或完整训练的逐 seed 轨迹一致。功能回归、真实数据集成与 benchmark 成绩分别记录。

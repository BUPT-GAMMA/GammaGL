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
| 产物兼容 | 本地移植须绑定其输入约定 | protocol/schema 4、cache 5；checkpoint 只检查 schema | protocol/schema 5、cache 6；缓存配置和指纹复检；正式 checkpoint 在载权重前检查具体 tokenizer 指纹 |

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

## 剩余差异与验证边界

- GammaGL 保留仅训练 split 拟合频率／BPE／词表；官方 UDI／prepare 链会收集全部数据。
  fixtures 显式给两边相同小语料，证明相同语料下协议相等，不能据此声称真实 benchmark
  的全数据统计或完整训练完全相同。训练范围不同可能产生不同频率排序和规则。
- 为训练集拟合支持 held-out 输入，未见原始 token 使用独立 OOV sentinel，避免冒充数值
  相同的 merge ID，最终进入 UNK；普通 0 依官方永久 alias 单独保留。
- GammaGL 词表采用上述显式双射；独立 GraphBPE 的 vocab_size 表示 ID 上界，官方引擎
  的同名统计表示词汇数量，不用它判断规则或模型输入是否相同。
- 仅支持简单无向语义和标量 token；不扩展官方实现可接受的多维 token／多重图范围。
  GammaGL 超长输入明确拒绝，增强后超长回退原输入；官方词表编码器会截断。
- 本次未验证 GPU、真实 release bundle 或正式 GTE 权重初始化，没有重跑六组 benchmark。
  旧成绩保持原记录，不能证明修复后的代码达到相同结果。

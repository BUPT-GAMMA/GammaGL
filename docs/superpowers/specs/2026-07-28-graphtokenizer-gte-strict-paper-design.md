# GraphTokenizer GT+GTE Strict Paper Benchmark Design

## Goal

在保持 GammaGL 现有目录结构和公共接口不变的前提下，按用户给定的论文规格，对
QM9、OGBG-molhiv 和 Peptides-struct 运行 GT+GTE 五次独立正式实验，并输出各数据集
主要指标的平均值和标准差。

## Scope

- 只运行 GT+GTE，不运行 GT+BERT。
- QM9 是包含 16 个输出的联合回归；不再按 `target_property` 单独训练。
- OGBG-molhiv 是单目标二分类，使用 OGB 官方 ROC-AUC evaluator。
- Peptides-struct 是 11 目标回归，使用逐任务 MAE 的平均值。
- 三个数据集均使用官方固定 split、Feuler、训练集独立 BPE、2000 次合并。
- 每个数据集运行五次，基础种子 42，具体采用 42、43、44、45、46，并在结果清单中
  标明这是论文未明确给出五个种子时的复现选择。

## Runtime Isolation

新环境位于 `/local/wrq/gammagl-graphtokenizer-paper-cu121`。它使用 Python 3.10、
PyTorch 2.1.2 CUDA 12.1、DGL 2.4.0 CUDA 12.1 和 PyTorch Geometric 2.4.0。
现有 `/local/wrq/gammagl-graphtokenizer` 不修改，作为已验证的回退环境。

所有缓存、临时文件、数据和输出都放在 `/local/wrq`：

- HuggingFace: `/local/wrq/cache/graphtokenizer-paper/huggingface`
- Torch: `/local/wrq/cache/graphtokenizer-paper/torch`
- XDG: `/local/wrq/cache/graphtokenizer-paper/xdg`
- 临时目录: `/local/wrq/tmp/graphtokenizer-paper`
- 数据: `/local/wrq/graph-tokenizer-data`
- 小测: `/local/wrq/graph-tokenizer-paper-small-20260728`
- 正式结果: `/local/wrq/graph-tokenizer-paper-formal-20260728`

## Protocol Corrections

`paper_protocol.py` 的 QM9 标签准备改为对 16 个训练目标分别计算均值和总体标准差，
分别归一化；验证和测试评估时按向量恢复原始单位，再对全部样本和全部目标计算 MAE。
任何维度不一致或非有限训练标签都必须在训练前失败。

三个 paper preset 的 `serialization` 均固定为 `feuler`，最大序列长度均为 8192。
不截断超长序列；任何序列超过 8192 时中止并明确报告，因为论文没有给出截断策略。

GTE 从 `Alibaba-NLP/gte-multilingual-base` 的已缓存远程配置创建并随机初始化，随后在
构造模型前显式验证/覆盖：12 层、hidden 768、12 个注意力头、FFN 3072、GELU、hidden
dropout 0.1、attention dropout 0.1、RoPE、最大长度 8192、LayerNorm epsilon 1e-12。
模型清单记录全部这些字段和实际参数量，配置不符时在训练前失败。

## Test Gates

正式训练必须依次通过以下门槛：

1. 单元/回归测试：QM9 16 目标归一化与还原、QM9 MAE、统一 Feuler、GTE 结构字段、
   强制至少一个 MLM mask、有限值检查和五次聚合。
2. 严格环境审计：精确包版本、CUDA 可用、C++ BPE 可导入、三个官方 split 的数量、
   无交叉、标签维度与有限性均正确。
3. 三个完整官方 split 的小型训练：每个数据集一个种子，MLM 1 epoch、微调 1 epoch，
   使用正式 batch size、2000 BPE merges 和 GTE 结构；同时监控显存、温度、NaN 和输出。
4. 只有三个小型训练全部退出码为零、产生有限指标和可恢复 checkpoint，才启动正式
   tmux 会话。

## Formal Execution

使用 GPU 3、4、5，每张卡一个数据集。正式配置为 MLM 200 epochs、微调最多 200
epochs、AdamW、weight decay 0.1、论文指定学习率/warmup/梯度裁剪/mask probability、
验证集 early stopping patience 20。每个 epoch 只评估验证集；恢复验证最佳 checkpoint
后仅评估一次测试集。

三个 tmux 会话分别写独立日志、状态文件和 JSON 结果。结果聚合采用五次运行的算术
平均和总体标准差（`ddof=0`），并同时保留五个单次结果以便复核。

## Failure Policy

依赖安装、数据审计、小测或正式任务出现错误时，不继续启动后续门槛。先保存完整日志，
按 systematic-debugging 定位根因，写失败回归测试，完成修复并重新通过受影响门槛。
不得把失败、NaN、缺少某次 seed 或未完成任务汇总成正式性能结果。

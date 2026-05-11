# gammagl 中 RGT 整理等价性审查报告

> 审查对象: `d:\GammaGL-main\gammagl\` 下的 RGT 相关代码 vs `d:\GammaGL-main\RGT-main\` 原始重构代码
> 审查日期: 2026-05-06

---

## 一、gammagl 中 RGT 相关文件清单

| 文件路径 | 功能描述 |
|---------|---------|
| `gammagl/models/rgt.py` | RGT 主模型 |
| `gammagl/models/rgt_heads.py` | 下游任务 Heads (NC/LP/GC/Shot) |
| `gammagl/layers/conv/rgt_layers.py` | 常曲率线性层、聚合层、编码器 |
| `gammagl/layers/attention/rgt_attention.py` | 结构学习器、跨流形注意力、欧几里得注意力 |
| `gammagl/layers/manifolds/lorentz.py` | Lorentz 双曲流形 |
| `gammagl/layers/manifolds/sphere.py` | Sphere 球面流形 |
| `gammagl/layers/manifolds/euclidean.py` | Euclidean 欧几里得流形 |
| `gammagl/layers/manifolds/product.py` | Product Space 乘积流形 |
| `gammagl/layers/vector_quantize/vq_euclidean.py` | 欧几里得 VQ |
| `gammagl/layers/vector_quantize/vq_riemann.py` | 黎曼 VQ |
| `gammagl/utils/rgt_utils.py` | 数据加载工具 (实际是 datasets.py 的副本) |
| `gammagl/models/__init__.py` | 模型导出列表 |

---

## 二、架构级别的重大差异

### 2.1 RGT 主模型: 两个完全不同的架构

| 维度 | RGT-main `modules/model.py` | gammagl `models/rgt.py` |
|------|---------------------------|------------------------|
| **层数** | `n_layers` 个 StructuralBlock 循环 | **单层**（无循环） |
| **初始化** | InitBlock (三层并行编码) | euc_enc/hyp_enc/sph_enc 直接调用 |
| **结构学习** | 每层 Hyp_learner + Sph_learner + Euc_learner + proj 融合 | 单次 strutcture learner |
| **VQ** | VQBlock 统一管理三个VQ | 分别 `self.vq_H/S/E` 直接调用 |
| **损失** | 内置 `.loss()` 方法（含对比学习损失） | **无 `.loss()` 方法** |
| **forward 返回** | `(x_E, x_H, x_S, q_E, q_H, q_S, commit_loss_E, commit_loss_H, commit_loss_S)` 共9个 | `(final_z, final_z_original, final_z_logits, (ids), (vq_losses))` 共5个 |
| **输入数据格式** | `data.n_id`, `data.tokens`, `data.batch_tree/cycle/sequence` | `batch.batch_tree/cycle/sequence` 或 tuple格式 |

**结论**: gammagl 的 RGT 是一个**简化版**，缺少RGT-main的多块堆叠结构和对比学习损失。如果作为算法库的不同实现变体可以接受，但**不能声称与 RGT-main 等价**。

### 2.2 Heads 模块: 关键功能缺失

| 组件 | RGT-main `modules/heads.py` | gammagl `models/rgt_heads.py` | 差异 |
|------|---------------------------|------------------------------|------|
| **NodeClsHead** | 嵌入 pretrained RGT 模型 → logmap0 → concat data.x → **2层GCN** | 单独 **Linear 层** | **缺失 GCN** |
| **LinkPredHead** | pretrained RGT → logmap0 → concat → Linear → cosine_similarity | Linear → gather → reduce_sum product | 逻辑相似 |
| **GraphClsHead** | pretrained RGT → logmap0 → concat → **unsorted_segment_mean** → Linear | Sequential(Linear-ReLU-Linear) **无池化** | **缺失图池化** |
| **ShotNCHead** | 完整的 pretrained RGT → logmap0 → GCN → cosine_sim | `forward()` 是 **`pass`**！ | **完全未实现** |

---

## 三、已发现的 Bug 详细清单

### Bug 1 [致命]: `ShotNCHead.forward()` 是空实现

**文件**: `gammagl/models/rgt_heads.py` 第 103-104 行

```python
def forward(self, x, edge_index, train_mask=None, val_mask=None, test_mask=None):
    pass
```

**影响**: 调用此 head 时不会返回任何值，直接导致下游代码崩溃。

**修复建议**: 实现完整的前向传播逻辑，包括 backbone 训练和 cosine similarity 计算。

---

### Bug 2 [严重]: `GraphClsHead` 缺失图级池化

**文件**: `gammagl/models/rgt_heads.py` 第 70-72 行

```python
# gammagl 当前实现 - 错误!
def forward(self, x, batch=None):
    x = self.classifier(x)      # 对每个节点独立分类，不是图级分类
    return x

# RGT-main 正确实现 - 需要池化:
def forward(self, data):
    x = ...
    x = unsorted_segment_mean(x, batch, ...)   # 先池化到图级
    return self.head(self.drop(x))
```

**影响**: 图分类任务的输出维度错误，结果为节点级而非图级。

---

### Bug 3 [严重]: `NodeClsHead` 只是单一 Linear 层

**文件**: `gammagl/models/rgt_heads.py` 第 34 行

```python
# gammagl 当前 - 只是线性层
self.classifier = tlx.layers.Linear(in_features=..., out_features=args.num_classes)

# RGT-main - 使用2层 GCN
self.head = GCN(2, in_dim, hidden_dim, num_cls, drop_edge, drop_feats)
```

**影响**: 节点分类缺少邻居信息聚合能力，准确率预期显著低于原始实现。

---

### Bug 4 [中等]: 各 Head 未集成 pretrained RGT 模型

**文件**: `gammagl/models/rgt_heads.py`

在 RGT-main 中，所有 Head 都接收 `pretrained_model` 参数，在 forward 中先调用 `self.pretrained_model(data)` 获取编码特征，再传给分类器。而 gammagl 的 Head 直接接收原始特征 `x` 作为输入，**缺失了整个 RGT 编码环节**。

```python
# RGT-main NodeClsHead.forward:
def forward(self, data):
    x_E, x_H, x_S, q_E, q_H, q_S, ... = self.pretrained_model(data)  # RGT编码
    x = tlx.concat([data.x, logmap0(q_E/H/S)], axis=-1)
    return self.head(x, edge_index, num_nodes)

# gammagl NodeClsHead.forward:
def forward(self, x, y=None):     # 直接拿原始x，跳过了RGT
    x = self.classifier(self.dropout(x))
    return x
```

---

### Bug 5 [低]: `ConstCurveAgg.neg_dist` lambda 定义错误

**文件**: `gammagl/layers/conv/rgt_layers.py` 第 98-99 行

```python
self.neg_dist = lambda x, y: 2 + 2 * manifold.cinner(x, y) if isinstance(manifold, Lorentz) \
    else lambda x, y: -manifold.dist(x, y) ** 2
```

Python 解析为:
```python
lambda x, y: (2 + 2 * manifold.cinner(x, y) if isinstance(manifold, Lorentz) 
              else lambda x, y: -manifold.dist(x, y) ** 2)
```

当 manifold 不是 Lorentz 时，`self.neg_dist(x, y)` 返回的是一个 **lambda 函数对象**，而不是数值！

**影响**: 目前 `self.neg_dist` 在 gammagl 代码中未被调用，属于死代码，但一旦使用会引发类型错误。

---

### Bug 6 [低]: `gammagl/utils/rgt_utils.py` 文件名与内容不符

该文件实际包含的是完整的数据集加载代码 (`datasets.py`)，但放在 `utils/` 目录下命名 `rgt_utils.py`，容易造成混淆。建议将其移至 `data/datasets.py` 或重命名。

---

### 缺失项 7: 无 RGT 示例文件

`gammagl/examples/` 目录下不存在任何 RGT 相关的示例代码（如 `rgt_pretrain.py`、`rgt_node_classification.py` 等）。

---

## 四、等价性评估总表

| 模块 | 文件 | 等价状态 | 关键问题 |
|------|------|---------|---------|
| **Manifolds** | `layers/manifolds/*` | ✅ 等价 | 继承 geoopt + TLX 扩展，数学正确 |
| **VQ** | `layers/vector_quantize/*` | ✅ 等价 | |
| **Layer层** | `layers/conv/rgt_layers.py` | ✅ 基本等价 | `neg_dist` 死代码 bug (低) |
| **结构学习器** | `layers/attention/rgt_attention.py` | ✅ 等价 | |
| **RGT主模型** | `models/rgt.py` | ⚠️ **不等价** | 缺少多块循环、对比学习损失 |
| **Heads** | `models/rgt_heads.py` | ❌ **不等价** | 3个致命bug |
| **数据加载** | `utils/rgt_utils.py` | ✅ 等价 | 文件位置不当 |

---

## 五、修复优先级建议

| 优先级 | Bug | 修复方案 |
|--------|-----|---------|
| **P0** | ShotNCHead.forward = pass | 实现完整的前向传播 |
| **P0** | GraphClsHead 无池化 | 添加 `unsorted_segment_mean` |
| **P1** | NodeClsHead 缺失 GCN | 替换为2层GCN |
| **P1** | Heads 未集成 pretrained RGT | 添加 RGT 编码环节 |
| **P2** | RGT 主模型架构差异 | 明确这是简化版还是需要对齐 |
| **P3** | neg_dist lambda | 删除死代码或修复逻辑 |
| **P3** | rgt_utils.py 文件位置 | 移至 `data/` 目录 |

---

*审查报告完毕*

import os
from gammagl.datasets import MoleculeNet
from GammaGL.examples.graphormer.model import Graphormer
from tensorlayerx.dataflow import Subset
from gammagl.loader import DataLoader
from sklearn.model_selection import train_test_split
import tensorlayerx as tlx
from gammagl.layers.pool import global_mean_pool
from tqdm import tqdm
from GammaGL.gammagl.utils import tfunction
from tensorlayerx.model import TrainOneStep, WithLoss
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataset = MoleculeNet(root='./', name='ESOL')
print(dataset)

model = Graphormer(
    num_layers=3,
    input_node_dim=dataset.num_node_features,
    node_dim=128,
    input_edge_dim=dataset.num_edge_features,
    edge_dim=128,
    output_dim=dataset[0].y.shape[1],
    n_heads=4,
    max_in_degree=5,
    max_out_degree=5,
    max_path_distance=5,
)

class loss_fn(WithLoss):
    def __init__(self, net, loss_fn):
        super(loss_fn, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data):
        logits = self.backbone_network(data)
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

test_ids, train_ids = train_test_split([i for i in range(len(dataset))], test_size=0.8, random_state=42)
train_loader = DataLoader(Subset(dataset, train_ids), batch_size=64)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=64)

# optimizer = CustomOptimizer(lr=3e-4)
optimizer = tlx.optimizers.Adam(lr=3e-4)
# loss_function = tlx.losses.absolute_difference_error
loss_func = loss_fn(net=model, loss_fn=tlx.losses.absolute_difference_error)
train_one_step = TrainOneStep(loss_func=loss_func, optimizer=optimizer, train_weights=model.trainable_weights)


DEVICE = "cuda"


model.to(DEVICE)

# 训练循环
# 训练循环，迭代10次
for epoch in range(10):
    # 将模型设置为训练模式
    model.set_train()

    # 初始化批次损失
    batch_loss = 0.0

    # 遍历训练集 DataLoader
    for batch in tqdm(train_loader):
        # 将数据移动到指定的设备上
        # batch.to(DEVICE)

        # 获取标签数据
        y = batch.y

        # 清零梯度
        optimizer.zero_grad()

        # 计算模型输出
        output = global_mean_pool(model(batch), batch.batch)

        # 计算损失值
        loss = loss_function(output, y)

        # 累加批次损失
        batch_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        tfunction.clip_grad_norm(model.parameters(), 1.0)

        # 参数更新
        optimizer.step()

    # 打印训练集的平均损失
    print("TRAIN_LOSS", batch_loss / len(train_ids))

    # 将模型设置为评估模式
    model.set_eval()

    # 初始化批次损失
    batch_loss = 0.0

    # 遍历测试集 DataLoader
    for batch in tqdm(test_loader):
        # 将数据移动到指定的设备上
        # batch.to(DEVICE)

        # 获取标签数据
        y = batch.y

        # 在评估模式下，不进行梯度计算
        with tfunction.no_grad():
            # 计算模型输出
            output = global_mean_pool(model(batch), batch.batch)

            # 计算损失值
            loss = loss_function(output, y)

        # 累加批次损失
        batch_loss += loss.item()

    # 打印测试集的平均损失
    print("EVAL LOSS", batch_loss / len(test_ids))

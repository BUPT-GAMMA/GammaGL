import tensorlayerx as tlx
from tensorlayerx import nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        # 使用线性层来模拟双线性操作
        self.fc1 = nn.Linear(
            in_features=n_h, 
            out_features=n_h, 
            W_init=tlx.initializers.XavierUniform(),
            b_init=tlx.initializers.Zeros()
        )
        # 不预先定义投影层，而是在第一次前向传播时创建
        self.proj = None
        self.c_dim = None
        self.h_dim = None

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # 获取输入维度
        c_dim = tlx.get_tensor_shape(c)[-1]
        h_dim = tlx.get_tensor_shape(h_pl)[-1]
        batch_size = tlx.get_tensor_shape(h_pl)[0]
        
        # 如果c和h_pl的特征维度不同，并且投影层尚未创建或维度发生变化，创建一个投影层
        if c_dim != h_dim and (self.proj is None or c_dim != self.c_dim or h_dim != self.h_dim):
            self.c_dim = c_dim
            self.h_dim = h_dim
            self.proj = nn.Linear(
                in_features=c_dim, 
                out_features=h_dim, 
                W_init=tlx.initializers.XavierUniform(),
                b_init=tlx.initializers.Zeros()
            )
        
        # 如果需要投影
        if c_dim != h_dim:
            c_proj = self.proj(c)
        else:
            c_proj = c
        
        # 将c_proj转换为与h_pl相同的批次大小
        c_x = tlx.expand_dims(c_proj, 0)
        c_x = tlx.tile(c_x, [batch_size, 1])
        
        # 处理正样本 - 使用元素级乘法和求和
        h_pl_proj = self.fc1(h_pl)  # [batch_size, n_h]
        # 计算内积
        sc_1 = tlx.reduce_sum(h_pl_proj * c_x, axis=1)  # [batch_size]
        
        # 处理负样本
        h_mi_proj = self.fc1(h_mi)  # [batch_size, n_h]
        # 计算内积
        sc_2 = tlx.reduce_sum(h_mi_proj * c_x, axis=1)  # [batch_size]
        
        # 添加偏置（如果有）
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        
        # 连接得分
        logits = tlx.concat([sc_1, sc_2], axis=0)
        
        return logits

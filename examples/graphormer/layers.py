from typing import Tuple
from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
from gammagl.utils.platform_utils import Tensor
import gammagl
from ..graphormer.utils import decrease_to_max_value
from ...gammagl.utils import tfunction


class CentralityEncoding(tlx.nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: 节点的最大入度
        :param max_out_degree: 节点的最大出度
        :param node_dim: 节点特征的隐藏维度
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = tlx.nn.Parameter(tlx.ops.random_normal((max_in_degree, node_dim)))
        self.z_out = tlx.nn.Parameter(tlx.ops.random_normal((max_out_degree, node_dim)))

    def forward(self, x: Tensor, edge_index) -> Tensor:
        """
        :param x: 节点特征矩阵
        :param edge_index: 图的边索引（邻接列表）
        :return: torch.Tensor，经过中心性编码后的节点嵌入
        """
        num_nodes = x.shape[0]

        # 计算每个节点的入度和出度，并将其限制在最大值以内
        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(), self.max_in_degree)
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(), self.max_out_degree)

        # 将中心性编码应用于节点特征
        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: 节点之间的最大路径距离
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        # 参数矩阵b，用于空间编码
        self.b = nn.Parameter(tlx.random_normal(tuple([self.max_path_distance])))

    def forward(self, x: gammagl.utils.platform_utils.Tensor, paths) -> gammagl.utils.platform_utils.Tensor:
        """
        :param x: 节点特征矩阵
        :param paths: 两两节点路径
        :return: torch.Tensor，空间编码矩阵
        """
        # 初始化空间编码矩阵
        spatial_matrix = tfunction.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)
        for src in paths:
            for dst in paths[src]:
                # 通过路径长度确定b的权重
                spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1]

        return spatial_matrix


def dot_product(x1, x2) -> gammagl.utils.platform_utils.Tensor:
    return (x1 * x2).sum(dim=1)


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: 边特征矩阵的维度数
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance

        # 边向量参数
        self.edge_vector = nn.Parameter(tlx.random_normal((self.max_path_distance, self.edge_dim)))

    def forward(self, x: gammagl.utils.platform_utils.Tensor, edge_attr: gammagl.utils.platform_utils.Tensor, edge_paths) -> gammagl.utils.platform_utils.Tensor:
        """
        :param x: 节点特征矩阵
        :param edge_attr: 边特征矩阵
        :param edge_paths: 边索引中的两两节点路径
        :return: torch.Tensor，边编码矩阵
        """
        # 初始化边编码矩阵
        cij = tfunction.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                # 获取路径中的前max_path_distance个节点
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                # 计算边编码的加权平均值
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        # 处理NaN值，将其替换为0
        cij = tfunction.nan_to_num(cij)
        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: 节点特征矩阵输入的维度
        :param dim_q: 查询节点特征矩阵输入的维度
        :param dim_k: 键节点特征矩阵输入的维度
        :param edge_dim: 边特征矩阵的维度数
        """
        super().__init__()

        # 边编码模块
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        # 线性变换层
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                query: gammagl.utils.platform_utils.Tensor,
                key: gammagl.utils.platform_utils.Tensor,
                value: gammagl.utils.platform_utils.Tensor,
                edge_attr: gammagl.utils.platform_utils.Tensor,
                b: gammagl.utils.platform_utils.Tensor,
                edge_paths,
                ptr) -> gammagl.utils.platform_utils.Tensor:
        """
        :param query: 节点特征矩阵
        :param key: 节点特征矩阵
        :param value: 节点特征矩阵
        :param edge_attr: 边特征矩阵
        :param b: 空间编码矩阵
        :param edge_paths: 边索引中的两两节点路径
        :param ptr: 批次中显示图索引的批次指针
        :return: torch.Tensor，经过注意力操作后的节点嵌入
        """
        batch_mask_neg_inf = tfunction.full(size=(query.shape[0], query.shape[0]), fill_value=-1e6).to(
            next(self.parameters()).device)
        batch_mask_zeros = tfunction.zeros(size=(query.shape[0], query.shape[0])).to(next(self.parameters()).device)

        # 优化：去掉切片，改用torch操作
        if type(ptr) == type(None

                             ):
            # 如果没有批次指针，表示只有一个图，将mask设置为1
            batch_mask_neg_inf = tfunction.ones(size=(query.shape[0], query.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            # 根据批次指针设置mask
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        # 线性变换
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        # 边编码
        c = self.edge_encoding(query, edge_attr, edge_paths)
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        a = (a + b + c) * batch_mask_neg_inf
        softmax = tfunction.softmax(input_array=a, axis=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x


# FIX: 使用稀疏注意力代替普通注意力，因为GNNs的特殊性（批次图中的所有节点将交换注意力）
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: 注意力头的数量
        :param dim_in: 节点特征矩阵输入的维度
        :param dim_q: 查询节点特征矩阵输入的维度
        :param dim_k: 键节点特征矩阵输入的维度
        :param edge_dim: 边特征矩阵的维度数
        """
        super().__init__()

        # 多头注意力机制
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )

        # 线性变换层
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: gammagl.utils.platform_utils.Tensor,
                edge_attr: gammagl.utils.platform_utils.Tensor,
                b: gammagl.utils.platform_utils.Tensor,
                edge_paths,
                ptr) -> gammagl.utils.platform_utils.Tensor:
        """
        :param x: 节点特征矩阵
        :param edge_attr: 边特征矩阵
        :param b: 空间编码矩阵
        :param edge_paths: 边索引中的两两节点路径
        :param ptr: 批次中显示图索引的批次指针
        :return: torch.Tensor，经过所有注意力头后的节点嵌入
        """
        return self.linear(
            tfunction.concatenate_tensor([
                attention_head(x, x, x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], axis=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, max_path_distance):
        """
        :param node_dim: 节点特征矩阵输入的维度
        :param edge_dim: 边特征矩阵输入的维度
        :param n_heads: 注意力头的数量
        :param max_path_distance: 最大路径距离
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads

        # 注意力层
        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )

        # 层归一化层
        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_2 = nn.LayerNorm(node_dim)

        # 前馈神经网络
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self,
                x: gammagl.utils.platform_utils.Tensor,
                edge_attr: gammagl.utils.platform_utils.Tensor,
                b: gammagl.utils.platform_utils.Tensor,
                edge_paths,
                ptr) -> Tuple[gammagl.utils.platform_utils.Tensor, gammagl.utils.platform_utils.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: 节点特征矩阵
        :param edge_attr: 边特征矩阵
        :param b: 空间编码矩阵
        :param edge_paths: 边索引中的两两节点路径
        :param ptr: 批次中显示图索引的批次指针
        :return: torch.Tensor，经过Graphormer层操作后的节点嵌入
        """
        # 多头注意力层
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x

        # 前馈神经网络层
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new

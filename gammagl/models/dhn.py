import tensorlayerx as tlx

type2idx = {
    'M': 0,
    'A': 1,
    # 'C': 2,
    # 'T': 3
}

NODE_TYPE = len(type2idx)
K_HOP = 2

NUM_FEA = (K_HOP + 2) * 4 + NODE_TYPE
NUM_NEIGHBOR = 5


class DHN(tlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = tlx.nn.Linear(in_features=36, out_features=64, act=tlx.nn.ELU(), W_init="xavier_uniform")
        self.lin2 = tlx.nn.Linear(in_features=82, out_features=64, act=tlx.nn.ELU(), W_init="xavier_uniform")
        self.lin3 = tlx.nn.Linear(in_features=64, out_features=64, act=tlx.nn.ELU(), W_init="xavier_uniform")

    def forward(self, fea):
        node = tlx.convert_to_tensor(fea[:, :NUM_FEA])

        # 提取neigh1和neigh2
        neigh1 = tlx.convert_to_tensor(fea[:, NUM_FEA:NUM_FEA * (NUM_NEIGHBOR + 1)])
        neigh1 = tlx.reshape(neigh1, [-1, NUM_NEIGHBOR, NUM_FEA])

        neigh2 = tlx.convert_to_tensor(fea[:, NUM_FEA * (NUM_NEIGHBOR + 1):])
        neigh2 = tlx.reshape(neigh2, [-1, NUM_NEIGHBOR, NUM_NEIGHBOR, NUM_FEA])

        # 对neigh2进行聚合
        neigh2_agg = tlx.reduce_mean(neigh2, axis=2)

        # 连接node和聚合后的neigh2
        tmp = tlx.concat(
            [neigh1, neigh2_agg],
            axis=2
        )

        # 将tmp重塑为二维张量
        flattened_tmp = tmp.view(64 * 5, 36)  

        # 将二维张量传递给线性层
        flattened_output = self.lin1(flattened_tmp)  

         # 将输出张量重新整形为三维形状
        tmp = flattened_output.view(64, 5, 64)  

        # 对node和tmp进行聚合
        emb = tlx.concat(
            [
                node, tlx.reduce_mean(tmp, axis=1)
            ],
            axis=1
        )

        # 通过其他线性层
        emb = self.lin2(emb)
        emb = self.lin3(emb)

        return emb


class DHNModel(tlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.dhn1 = DHN()
        self.dhn2 = DHN()
        self.lin1 = tlx.nn.Linear(in_features=128, out_features=32, act=tlx.nn.ELU(), W_init="xavier_uniform")
        self.lin2 = tlx.nn.Linear(in_features=32, out_features=1, act=tlx.nn.ELU(), W_init="xavier_uniform")

    def forward(self, n1, n2, label):
        n1_emb = self.dhn1(n1)
        n2_emb = self.dhn2(n2)

        pred = self.lin1(tlx.concat([n1_emb, n2_emb], axis=1))
        pred = self.lin2(pred)

        return pred

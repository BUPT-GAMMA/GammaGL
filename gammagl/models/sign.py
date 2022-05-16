import tensorlayerx as tlx


class SignModel(tlx.nn.Module):
    def __init__(self, K, in_feat, hid_feat, num_classes, drop):
        super(SignModel, self).__init__()
        self.act = tlx.ReLU()
        self.drop = tlx.nn.Dropout(drop)

        # he_normal is kaiming_normal
        # init = tlx.initializers.he_normal()
        cur = []
        for _ in range(K + 1):
            cur.append(tlx.nn.Linear(in_features=in_feat, out_features=hid_feat))
        self.lin = tlx.nn.Linear(in_features=(K + 1) * hid_feat, out_features=num_classes)
        self.lins = tlx.nn.Sequential(cur)

    def forward(self, xs):
        hs = []
        for x, lin in zip(xs, self.lins):
            h = lin(x)
            h = self.drop(self.act(h))
            hs.append(h)
        h = tlx.concat(hs, axis=-1)
        h = self.lin(h)
        return h




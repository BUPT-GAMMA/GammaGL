import tensorlayerx as tlx

class SignModel(tlx.nn.Module):
    def __init__(self, K, in_feat, hid_feat, num_classes, drop):
        super(SignModel, self).__init__()
        self.act = tlx.ReLU()
        self.drop = tlx.nn.Dropout(drop)
        self.lins = tlx.nn.SequentialLayer()
        # he_normal is kaiming_normal
        # init = tlx.initializers.he_normal()
        for _ in range(K + 1):
            self.lins.append(tlx.nn.Linear(in_features=in_feat, out_features=hid_feat))
        self.lin = tlx.nn.Linear(in_features=(K + 1) * hid_feat, out_features=num_classes)
    def forward(self, xs):
        hs = []
        for x, lin in zip(xs, self.lins):
            h = lin(x)
            h = self.drop(self.act(h))
            hs.append(h)
        h = tlx.concat(hs, axis=-1)
        h = self.lin(h)
        return h




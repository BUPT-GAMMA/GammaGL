from gammagl.datasets.ogb_node import OgbNodeDataset

def test():
    data=OgbNodeDataset('ogbn-arxiv')
    print(data[0])

test()
from gammagl.datasets.ogb_node import OgbNodeDataset

def test_ogbnodedataset():
    data=OgbNodeDataset('ogbn-arxiv')
    print(data[0])

test_ogbnodedataset()
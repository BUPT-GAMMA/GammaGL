from gammagl.datasets.ogb_node import OgbNodeDataset

def test_ogbnodedataset():
    data = OgbNodeDataset('ogbn-arxiv')
    split = data.get_idx_split()
    print(data[0])
    print(split)

test_ogbnodedataset()
from gammagl.datasets.ogb_link import OgbLinkDataset

def test_ogblinkdataset():
    data = OgbLinkDataset('ogbl-ppa')
    split = data.get_edge_split()
    print(data[0])
    print(split)

test_ogblinkdataset()
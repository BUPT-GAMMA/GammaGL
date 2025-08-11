from gammagl.datasets.ogb_link import OgbLinkDataset

def test_ogblinkdataset():
    data=OgbLinkDataset('ogbl-ppa')
    print(data[0])

test_ogblinkdataset()
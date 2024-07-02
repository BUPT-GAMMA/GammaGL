from gammagl.datasets.ogb_graph import OgbGraphDataset

def test_ogbgraphdataset():
    data=OgbGraphDataset('ogbg-molhiv')
    print(data)
    print(data[0])

test_ogbgraphdataset()
from gammagl.datasets.ogb_graph import OgbGraphDataset

def test_ogbgraphdataset():
    data=OgbGraphDataset('ogbg-molhiv')
    split = data.get_idx_split()
    print(data)
    print(data[0])
    print(split)

test_ogbgraphdataset()
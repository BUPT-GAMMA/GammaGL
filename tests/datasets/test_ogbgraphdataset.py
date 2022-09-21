from gammagl.datasets.ogb_graph import OgbGraphDataset

def test():
    data=OgbGraphDataset('ogbg-molhiv')
    print(data)
    print(data[0])

test()
from gammagl.datasets.ogb_link import OgbLinkDataset

def test():
    data=OgbLinkDataset('ogbl-ppa')
    print(data[0])

test()
from gammagl.datasets.hgb import HGBDataset

root = ''
def test_hgb_dataset():
    dataset_list = ['dblp_hgb', 'imdb_hgb', 'acm_hgb']
    # dataset_list = ['dblp_hgb', 'imdb_hgb', 'acm_hgb']
    for i in range(len(dataset_list)):
        heterograph = None
        dataset = HGBDataset(root=root, name=dataset_list[i])
        heterograph = dataset[0]
        assert heterograph is not None

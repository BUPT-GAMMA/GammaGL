import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.datasets.hgb import HGBDataset
import os.path as osp

root = ''
def test_hgb_dataset():
    dataset_list = ['dblp_hgb', 'imdb_hgb', 'acm_hgb']
    for i in range(len(dataset_list)):
        heterograph = None
        dataset = HGBDataset(root=root, name=dataset_list[i])
        heterograph = dataset[0]
        assert heterograph is not None

import torch.nn.functional as F
import numpy as np
import os.path as osp
from yaml import SafeLoader
from get_data import get_data
from data_utils import rand_train_test_idx

dataset = 'film'
path = osp.join(osp.expanduser('~'), 'datasets', dataset)

train_prop = 0.5
valid_prop = 0.25

data = get_data(path, dataset)
# get the splits for all runs
for i in range(10):
    ignore_negative = False if dataset == 'ogbn-proteins' else True
    train_mask, valid_mask, test_mask = rand_train_test_idx(
        data.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
    splits_file_path = '{}/{}'.format(path, dataset) + '_split_50_25_' + str(i)
    np.savez(splits_file_path, train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask)

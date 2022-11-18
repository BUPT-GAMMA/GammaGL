import os
import os.path as osp
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TL_BACKEND']='torch'
sys.path.append(os.path.abspath('.'))

import tensorlayerx as tlx
tlx.set_device()

from gammagl.datasets import Planetoid
path = osp.join(osp.abspath('..'),'data','Planetoid')
dataset = Planetoid(path, name='Cora')
data = dataset[0]

from gammagl.transforms import RandomLinkSplit
transform = RandomLinkSplit(is_undirected=True, split_labels=True)
train_, val_, test_ = transform(data)
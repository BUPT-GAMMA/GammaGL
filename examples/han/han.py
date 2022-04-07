
import os.path as osp
from typing import Dict, List, Union

import gammagl.transforms as T
from gammagl.datasets import IMDB
# from gammagl.nn import HANConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
metapaths = [[('movie', 'actor'), ('actor', 'movie')],
			 [('movie', 'director'), ('director', 'movie')]]
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
						   drop_unconnected_nodes=True)
dataset = IMDB(path, transform=transform)
data = dataset[0]
print(data)
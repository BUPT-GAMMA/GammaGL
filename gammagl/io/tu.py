import glob
import os
import os.path as osp

import numpy as np
from gammagl.io import read_txt_array
from gammagl.data import Graph

names = [
	'A', 'graph_indicator', 'node_labels', 'node_attributes'
	'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(folder, prefix):
	files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
	names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]
	
	edge_index = read_file(folder, prefix, 'A', np.int32).transpose() - 1
	batch = read_file(folder, prefix, 'graph_indicator', np.int32) - 1
	
	node_attributes = node_labels = None
	if 'node_attributes' in names:
		node_attributes = read_file(folder, prefix, 'node_attributes', dtype=np.float32)
	if 'node_labels' in names:
		node_labels = read_file(folder, prefix, 'node_labels', np.int32)
		if len(node_labels.shape) >= 1:
			# node_labels = np.expand_dims(node_labels, -1)
			node_labels = node_labels.squeeze()
		node_labels = node_labels - node_labels.min()
		node_labels = np.eye(node_labels.max() + 1)[node_labels]
	# node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
	x = cat([node_attributes, node_labels])
	
	edge_attributes, edge_labels = None, None
	if 'edge_attributes' in names:
		edge_attributes = read_file(folder, prefix, 'edge_attributes')
	if 'edge_labels' in names:
		edge_labels = read_file(folder, prefix, 'edge_labels', np.int32)
		if len(edge_labels.shape) >= 1:
			edge_labels = edge_labels.squeeze()
		edge_labels = edge_labels - edge_labels.min()
		edge_labels = np.eye(edge_labels.max() + 1)[edge_labels]
	edge_attr = cat([edge_attributes, edge_labels])
	
	y = None
	if 'graph_attributes' in names:  # Regression problem.
		y = read_file(folder, prefix, 'graph_attributes')
	elif 'graph_labels' in names:  # Classification problem.
		y = read_file(folder, prefix, 'graph_labels', np.int32)
		_, y = np.unique(y, return_inverse=True)
	
	num_nodes = edge_index.max().item() + 1 if x is None else x.shape[0]
	edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
	# edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
	#                                  num_nodes)
	
	# data = Graph(edge_index=edge_index, edge_feat=edge_attr, node_feat=x, node_label=y)
	graph, slices = split(edge_index, batch, x, edge_attr, y)
	
	return graph, slices


def remove_self_loops(edge_index, edge_attr=None):
	r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
	that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

	Parameters
	----------
	edge_index: LongTensor
		The edge indices.
	edge_attr: Tensor, optional
		Edge weights or multi-dimensional
		edge features. (default: :obj:`None`)

	:rtype: (:class:`LongTensor`, :class:`Tensor`)
	"""
	mask = edge_index[0] != edge_index[1]
	edge_index = edge_index[:, mask]
	if edge_attr is None:
		return edge_index, None
	else:
		return edge_index, edge_attr[mask]


def read_file(folder, prefix, name, dtype=None):
	path = osp.join(folder, f'{prefix}_{name}.txt')
	return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
	seq = [item for item in seq if item is not None]
	seq = [np.expand_dims(item, axis=-1) if len(item.shape) == 1 else item for item in seq]
	return np.concatenate(seq, axis=-1) if len(seq) > 0 else None


def split(edge_index, batch, x=None, edge_attr=None, y=None):
	node_slice = np.bincount(batch).cumsum()
	node_slice = np.concatenate([np.array([0]), node_slice])
	
	row, _ = edge_index
	edge_slice = np.bincount(batch[row]).cumsum()
	edge_slice = np.concatenate([np.array([0]), edge_slice])
	
	# Edge indices should start at zero for every graph.
	edge_index -= np.expand_dims(node_slice[batch[row]], axis=0)
	
	slices = {'edge_index': edge_slice}
	if x is not None:
		slices['x'] = node_slice
	# else:
	#     # Imitate `collate` functionality:
	#     num_nodes = torch.bincount(batch).tolist()
	#     num_nodes = batch.numel()
	if edge_attr is not None:
		slices['edge_attr'] = edge_slice
	if y is not None:
		if y.shape[0] == batch.shape[0]:
			slices['y'] = node_slice
		else:
			slices['y'] = np.arange(0, batch[-1] + 2, dtype=np.int32)
	graph = Graph(x=x, edge_index=edge_index, edge_feat=edge_attr, y=y).tensor()
	return graph, slices


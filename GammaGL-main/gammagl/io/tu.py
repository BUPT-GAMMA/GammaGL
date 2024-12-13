import glob
import os
import os.path as osp

import numpy as np
import tensorlayerx as tlx
from gammagl.io import read_txt_array
from gammagl.data import Graph
from gammagl.utils import coalesce, remove_self_loops

names = [
	'A', 'graph_indicator', 'node_labels', 'node_attributes'
	'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(folder, prefix):
	files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
	names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]
	
	edge_index = read_file(folder, prefix, 'A', np.int64).transpose() - 1
	batch = read_file(folder, prefix, 'graph_indicator', np.int64) - 1
	
	node_attributes = np.empty((batch.shape[0], 0), dtype=np.float32)
	if 'node_attributes' in names:
		node_attributes = read_file(folder, prefix, 'node_attributes', dtype=np.float32)

	node_labels = np.empty((batch.shape[0], 0), dtype=np.float32)
	if 'node_labels' in names:
		node_labels = read_file(folder, prefix, 'node_labels', np.int64)
		if len(node_labels.shape) == 1:
			# node_labels = np.expand_dims(node_labels, -1)
			node_labels = np.expand_dims(node_labels, axis=-1)
		node_labels = node_labels - node_labels.min(axis=0)[0]
		node_labels = np.eye(node_labels.max() + 1)[node_labels].squeeze()

	edge_attributes = np.empty((edge_index.shape[0], 0), dtype=np.float32)
	if 'edge_attributes' in names:
		edge_attributes = read_file(folder, prefix, 'edge_attributes')

	edge_labels = np.empty((edge_index.shape[1], 0))
	if 'edge_labels' in names:
		edge_labels = read_file(folder, prefix, 'edge_labels', np.int64)
		if len(edge_labels.shape) == 1:
			edge_labels = np.expand_dims(edge_labels, axis=-1)
		edge_labels = edge_labels - edge_labels.min()
		edge_labels = np.eye(edge_labels.max() + 1)[edge_labels]

	x = cat([node_attributes, node_labels])
	edge_attr = cat([edge_attributes, edge_labels])

	y = None
	if 'graph_attributes' in names:  # Regression problem.
		y = read_file(folder, prefix, 'graph_attributes')
	elif 'graph_labels' in names:  # Classification problem.
		y = read_file(folder, prefix, 'graph_labels', np.int32)
		_, y = np.unique(y, return_inverse=True)
	
	num_nodes = edge_index.max().item() + 1 if x is None else x.shape[0]
	edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
	if edge_attr is None:
		edge_index = coalesce(edge_index, num_nodes=num_nodes)
	else:
		edge_index, edge_attr = coalesce(edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

	edge_index = tlx.convert_to_numpy(edge_index)

	# data = Graph(edge_index=edge_index, edge_feat=edge_attr, node_feat=x, node_label=y)
	graph, slices = split(edge_index, batch, x, edge_attr, y)

	sizes = {
		'num_node_attributes': node_attributes.shape[-1],
		'num_node_labels': node_labels.shape[-1],
		'num_edge_attributes': edge_attributes.shape[-1],
		'num_edge_labels': edge_labels.shape[-1],
	}

	return graph, slices, sizes


def read_file(folder, prefix, name, dtype=None):
	path = osp.join(folder, f'{prefix}_{name}.txt')
	return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
	seq = [item for item in seq if item is not None]
	seq = [item for item in seq if item.size > 0]
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
	graph = Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).tensor()
	return graph, slices


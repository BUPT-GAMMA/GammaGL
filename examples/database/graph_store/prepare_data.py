from gammagl.utils import mask_to_index
from gammagl.datasets import Reddit
import tensorlayerx as tlx
import numpy as np
from gdbi.ggl import Neo4jFeatureStore, Neo4jGraphStore

dataset = Reddit('')
graph = dataset[0]

train_idx = tlx.convert_to_numpy(mask_to_index(graph.train_mask))
val_idx = tlx.convert_to_numpy(mask_to_index(graph.val_mask))
test_idx = tlx.convert_to_numpy(mask_to_index(graph.test_mask))

np.savez('idx.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

uri = 'bolt://localhost:7687'
user_name= 'neo4j'
password= 'neo4j'

feature_store = Neo4jFeatureStore(uri=uri, user_name=user_name, password=password)
graph_store = Neo4jGraphStore(uri=uri, user_name=user_name, password=password)

feature_store['reddit_node', 'x', tlx.arange(start=0, limit=graph.x.shape[0])] = graph.x
feature_store['reddit_node', 'y', tlx.arange(start=0, limit=graph.x.shape[0])] = graph.y
graph_store['reddit_edge', 'coo'] = graph.edge_index
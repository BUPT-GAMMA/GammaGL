# gammagl/transforms/nr_augmentor.py

import tensorlayerx as tlx
import numpy as np
import random
from gammagl.data import Graph
from gammagl.utils import to_undirected, coalesce

class NR_Augmentor:
    def __init__(self, probability=0.5):
        """
        Neighbor Replacement Augmentor for GammaGL.
        This augmentor implements the NeighborReplace (NR) strategy from the paper
        "Regularizing Graph Neural Networks via Consistency-Diversity Graph Augmentations".

        Args:
            probability (float): Probability of replacing a 1-hop neighbor with a 2-hop neighbor.
        """
        self.probability = probability

    def _get_1hop_neighbors_dict(self, edge_index, num_nodes):
        """
        Creates an adjacency list dictionary {node_idx: [neighbor1, neighbor2,...]}
        from edge_index, containing unique 1-hop neighbors (excluding self-loops).
        """
        adj = {i: set() for i in range(num_nodes)}
        src_nodes, dst_nodes = tlx.convert_to_numpy(edge_index[0]), tlx.convert_to_numpy(edge_index[1])
        for i in range(len(src_nodes)):
            u, v = src_nodes[i], dst_nodes[i]
            if u != v:
                adj[u].add(v)
        return {node_idx: list(neighbors) for node_idx, neighbors in adj.items()}

    def __call__(self, original_graph: Graph) -> Graph:
        """
        Applies Neighbor Replacement augmentation to the input graph.
        Making the class callable is a common pattern for transforms.

        Args:
            original_graph (gammagl.data.Graph): The original graph object.

        Returns:
            gammagl.data.Graph: The augmented graph object.
        """
        num_nodes = original_graph.num_nodes
        
        adj_1hop_dict = self._get_1hop_neighbors_dict(original_graph.edge_index, num_nodes)

        new_edge_src_list = []
        new_edge_dst_list = []

        for u_node_idx in range(num_nodes):
            current_1hop_neighbors_of_u = adj_1hop_dict[u_node_idx]
            if not current_1hop_neighbors_of_u:
                continue

            for v_neighbor_idx in current_1hop_neighbors_of_u:
                if random.random() < self.probability:
                    potential_2hop_neighbors_of_u_via_v = [
                        vv_node for vv_node in adj_1hop_dict.get(v_neighbor_idx, [])
                        if vv_node != u_node_idx and vv_node != v_neighbor_idx
                    ]
                    
                    if potential_2hop_neighbors_of_u_via_v:
                        vv_chosen = random.choice(potential_2hop_neighbors_of_u_via_v)
                        new_edge_src_list.append(u_node_idx)
                        new_edge_dst_list.append(vv_chosen)
                    else:
                        new_edge_src_list.append(u_node_idx)
                        new_edge_dst_list.append(v_neighbor_idx)
                else:
                    new_edge_src_list.append(u_node_idx)
                    new_edge_dst_list.append(v_neighbor_idx)
        
        if not new_edge_src_list:
            aug_edge_index = tlx.convert_to_tensor(np.array([[],[]]), dtype=tlx.int64)
        else:
            temp_edge_index = tlx.stack([
                tlx.convert_to_tensor(new_edge_src_list, dtype=tlx.int64),
                tlx.convert_to_tensor(new_edge_dst_list, dtype=tlx.int64)
            ])
            
            undirected_temp_edge_index = to_undirected(temp_edge_index, num_nodes=num_nodes)
            aug_edge_index = coalesce(undirected_temp_edge_index, num_nodes=num_nodes)

        aug_graph = Graph(x=original_graph.x, edge_index=aug_edge_index, num_nodes=num_nodes)
        
        # Copy other essential attributes if they exist
        if hasattr(original_graph, 'y'): aug_graph.y = original_graph.y
        if hasattr(original_graph, 'train_mask'): aug_graph.train_mask = original_graph.train_mask
        if hasattr(original_graph, 'val_mask'): aug_graph.val_mask = original_graph.val_mask
        if hasattr(original_graph, 'test_mask'): aug_graph.test_mask = original_graph.test_mask
        
        return aug_graph
from ogb.nodeproppred import NodePropPredDataset
from gammagl.data import Graph
def ogbn_dataset(name, root = 'dataset', meta_dict = None):
    '''
        - name (str): name of the dataset
        - root (str): root directory to store the dataset folder
        - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                but when something is passed, it uses its information. Useful for debugging for external contributers.
    '''
    dataset=NodePropPredDataset(name=name,root=root,meta_dict=meta_dict)
    dataset=dataset[0]
    data = Graph(edge_index=dataset[0]['edge_index'], x=dataset[0]['node_feat'],y=dataset[1])
    data.num_nodes=dataset[0]['num_nodes']
    data.edge_attr = dataset[0]['edge_feat']
    data.tensor()
    return data
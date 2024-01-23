# Graph Database Interface Example



## Install gdbi
```bash
pip install git+https://github.com/xy-Ji/gdbi.git
```
gdbi implements 4 graph database interfaces. You can use these interfaces to operate the graph database and retrieve specified dataset from the graph database.

gdbi link: [https://github.com/xy-Ji/gdbi](https://github.com/xy-Ji/gdbi)

## Example
```python
from gdbi import NodeExportConfig, EdgeExportConfig, Neo4jInterface

node_export_config = list(NodeExportConfig(labelname, x_property_names, y_property_names))
edge_export_config = list(EdgeExportConfig(labelname, src_dst_label, x_property_names, y_property_names))
graph_database = Neo4jInterface()
conn = graph_database.GraphDBConnection(graph_address, user_name, password)
graph = graph_database.get_graph(conn, graph_name, node_export_config, edge_export_config)
```

Run example
```bash
TL_BACKEND=torch python cora_sage.py --dataset cora --n_epoch 500 --lr 0.005 --hidden_dim 512 --drop_rate 0.8
```
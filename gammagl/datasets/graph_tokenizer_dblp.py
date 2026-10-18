from ._graph_tokenizer_benchmark import PreprocessedGraphBenchmark


class GraphTokenizerDBLP(PreprocessedGraphBenchmark):
    r"""TU DBLP_v1 benchmark from the official GraphTokenizer release.

    Follows the `fixed official loader
    <https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/src/data/loader/dblp_loader.py>`_
    used in `"Graph Tokenization for Bridging Graphs and Transformers"
    <https://arxiv.org/abs/2603.11099>`_. Task: binary graph classification (two classes); label key: "label".
    This graph-classification dataset is distinct from the heterogeneous
    node-classification dataset :class:`DBLP`. Missing edge features use
    token 0; scalar node features are required.

    All constructor parameters and defaults, item shapes, split methods,
    required file layout and exceptions are documented in
    :class:`~gammagl.datasets._graph_tokenizer_benchmark.PreprocessedGraphBenchmark`.
    Each graph has y of shape (1, 1); use the supplied official split files.

    Examples
    --------
    .. code-block:: python

        from gammagl.datasets import GraphTokenizerDBLP

        # Requires the prepared release files under data/dblp/raw/.
        dataset = GraphTokenizerDBLP(root="data")
        train = dataset.get_split("train")
        graph = train[0]
    """

    name = 'dblp'
    display_name = 'DBLP_v1'
    aliases = ('dblp', 'dblp-v1', 'dblp_v1')
    task_type = 'binary_classification'
    num_tasks = 1
    num_classes = 2
    metric = 'accuracy'
    label_keys = ('label',)
    default_edge_token = 0

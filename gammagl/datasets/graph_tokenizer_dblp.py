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

    Parameters
    ----------
    root: str
        Dataset root. Prepared files are read from root/dblp/raw/:
        data.pkl (or data.pkl.gz), train_index.json, val_index.json and
        test_index.json. Construction can materialize a prepared local bundle;
        it does not download the official release. See
        :doc:`/notes/graph_tokenizer` for data preparation.
    transform: ``callable``
        Applied to each :class:`~gammagl.data.Graph` on access.
        (default: :obj:`None`)
    pre_transform: ``callable``
        Applied before saving processed graphs. Must preserve scalar tokens
        and official graph ordering. (default: :obj:`None`)
    pre_filter: ``callable``
        Must be None to preserve the official split indices.
        (default: :obj:`None`)
    force_reload: bool
        Reprocess existing raw data. (default: :obj:`False`)

    Raises
    ------
    FileNotFoundError
        If required raw files or a prepared local bundle are unavailable.
    ValueError
        If required scalar features or labels are invalid, split indices
        overlap or are out of range, directed metadata is present, or
        pre_filter is not None.

    Notes
    -----
    Indexing returns a :class:`~gammagl.data.Graph`: integer x (N,), edge_index
    (2, E), integer edge_attr (E,), and floating y (1, 1). Labels retain their
    original scale. Feature tokens follow the loader rules described in
    :class:`~gammagl.datasets._graph_tokenizer_benchmark.PreprocessedGraphBenchmark`.
    Pickles must be trusted; DGL release objects require the paper extra.
    Processed files are specific to the selected TLX backend.

    :meth:`~gammagl.datasets._graph_tokenizer_benchmark.PreprocessedGraphBenchmark.get_idx_split` returns copies of the official train / val / test
    indices. :meth:`~gammagl.datasets._graph_tokenizer_benchmark.PreprocessedGraphBenchmark.get_split` returns graphs in their fixed split order;
    filtering or generating a new random split changes the experiment.

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

from ._graph_tokenizer_benchmark import PreprocessedGraphBenchmark


class AQSOL(PreprocessedGraphBenchmark):
    r"""AQSOL solubility benchmark from the official GraphTokenizer release.

    Follows the `fixed official loader
    <https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/src/data/loader/aqsol_loader.py>`_
    used in `"Graph Tokenization for Bridging Graphs and Transformers"
    <https://arxiv.org/abs/2603.11099>`_. Task: regression; label key: "solubility".
    Predicts one continuous solubility target. Raw node features represent
    atomic numbers and raw edge features bond types; both are required.
    The dataset does not normalize targets; the trainer reports raw-scale MAE.

    Parameters
    ----------
    root: str
        Dataset root. Prepared files are read from root/aqsol/raw/:
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

        from gammagl.datasets import AQSOL

        # Requires the prepared release files under data/aqsol/raw/.
        dataset = AQSOL(root="data")
        train = dataset.get_split("train")
        graph = train[0]
    """

    name = 'aqsol'
    display_name = 'AQSOL'
    aliases = ('aqsol',)
    task_type = 'regression'
    num_tasks = 1
    metric = 'mae'
    label_keys = ('solubility',)

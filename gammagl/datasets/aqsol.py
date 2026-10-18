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

    All constructor parameters and defaults, item shapes, split methods,
    required file layout and exceptions are documented in
    :class:`~gammagl.datasets._graph_tokenizer_benchmark.PreprocessedGraphBenchmark`.
    Each graph has y of shape (1, 1); use the supplied official split files.

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

from ._graph_tokenizer_benchmark import PreprocessedGraphBenchmark


class Mutagenicity(PreprocessedGraphBenchmark):
    r"""TU Mutagenicity benchmark from the official GraphTokenizer release.

    Follows the `fixed official loader
    <https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/src/data/loader/mutagenicity_loader.py>`_
    used in `"Graph Tokenization for Bridging Graphs and Transformers"
    <https://arxiv.org/abs/2603.11099>`_. Task: binary graph classification (two classes); label key: "label".
    This is Mutagenicity, not the 188-graph MUTAG dataset. Missing edge
    features use token 0; scalar node features are required.

    All constructor parameters and defaults, item shapes, split methods,
    required file layout and exceptions are documented in
    :class:`~gammagl.datasets._graph_tokenizer_benchmark.PreprocessedGraphBenchmark`.
    Each graph has y of shape (1, 1); use the supplied official split files.

    Examples
    --------
    .. code-block:: python

        from gammagl.datasets import Mutagenicity

        # Requires the prepared release files under data/mutagenicity/raw/.
        dataset = Mutagenicity(root="data")
        train = dataset.get_split("train")
        graph = train[0]
    """

    name = 'mutagenicity'
    display_name = 'Mutagenicity'
    aliases = ('mutagenicity', 'mutag', 'muta')
    task_type = 'binary_classification'
    num_tasks = 1
    num_classes = 2
    metric = 'accuracy'
    label_keys = ('label',)
    default_edge_token = 0

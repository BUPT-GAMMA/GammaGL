from ._molecular_benchmark import PreprocessedMolecularBenchmark


class QM9(PreprocessedMolecularBenchmark):
    r"""The QM9 molecular property prediction benchmark.

    This loader follows the GammaGL :class:`InMemoryDataset` workflow. On the
    data is prepared by the GraphTokenizer single-process preparation command,
    which caches the graph pickle and official split files under ``raw``.
    """

    name = 'qm9'
    display_name = 'QM9'
    aliases = ('qm9',)
    task_type = 'regression'
    num_tasks = 16
    metric = 'mae'
    label_keys = (
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
        'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom',
        'h298_atom', 'g298_atom',
    )
    node_feature_columns = {'attr': 5, 'x': 0}

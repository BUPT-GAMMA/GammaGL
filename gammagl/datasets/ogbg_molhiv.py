from ._molecular_benchmark import PreprocessedMolecularBenchmark


class OGBGMolHIV(PreprocessedMolecularBenchmark):
    r"""The OGBG-molhiv molecular property prediction benchmark.

    On the first construction this loader downloads the released
    GraphTokenizer paper bundle and caches its preprocessed graphs and official
    OGB split files under ``raw``.
    """

    name = 'ogbg-molhiv'
    display_name = 'OGBG-molhiv'
    aliases = ('molhiv', 'ogbg-molhiv', 'ogbg_molhiv')
    task_type = 'binary_classification'
    num_tasks = 1
    metric = 'rocauc'
    label_keys = ('label',)

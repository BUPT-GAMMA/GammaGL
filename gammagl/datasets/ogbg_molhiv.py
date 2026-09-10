from ._molecular_benchmark import PreprocessedMolecularBenchmark


class OGBGMolHIV(PreprocessedMolecularBenchmark):
    r"""The OGBG-molhiv molecular property prediction benchmark.

    Data must be prepared by the GraphTokenizer single-process preparation
    command before construction; training never downloads the shared bundle.
    """

    name = 'ogbg-molhiv'
    display_name = 'OGBG-molhiv'
    aliases = ('molhiv', 'ogbg-molhiv', 'ogbg_molhiv')
    task_type = 'binary_classification'
    num_tasks = 1
    metric = 'rocauc'
    label_keys = ('label',)

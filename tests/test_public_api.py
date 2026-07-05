def test_layers_public_api_exports_message_passing():
    from gammagl.layers import MessagePassing
    from gammagl.layers.conv import MessagePassing as ConvMessagePassing

    assert MessagePassing is ConvMessagePassing


def test_models_public_api_all_is_importable():
    namespace = {}

    exec("from gammagl.models import *", namespace)

    import gammagl.models as models

    for name in models.__all__:
        assert name in namespace
        assert hasattr(models, name)


def test_datasets_public_api_all_is_importable():
    namespace = {}

    exec("from gammagl.datasets import *", namespace)

    import gammagl.datasets as datasets

    for name in datasets.__all__:
        assert name in namespace
        assert hasattr(datasets, name)


def test_utils_public_api_all_is_importable():
    namespace = {}

    exec("from gammagl.utils import *", namespace)

    import gammagl.utils as utils

    for name in utils.__all__:
        assert name in namespace
        assert hasattr(utils, name)

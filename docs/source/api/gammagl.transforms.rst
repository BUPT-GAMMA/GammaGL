gammagl.transforms
==========================

.. currentmodule:: gammagl.transforms

.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :template: autosummary/transforms_class.rst

    gammagl.transforms.BaseTransform
    gammagl.transforms.AddMetaPaths
    gammagl.transforms.Compose
    gammagl.transforms.SIGN
    gammagl.transforms.NormalizeFeatures
    gammagl.transforms.DropEdge
    gammagl.transforms.RandomLinkSplit
    gammagl.transforms.mask_test_edges
    gammagl.transforms.sparse_to_tuple
    FrequencyGuidedEulerianSerializer
    EulerianSerializer
    graph_serializer.GraphSerializationResult
    BPECodebook
    GraphBPE
    GraphTokenizer
    GraphTokenizerSpecialTokens
    GraphTokenizationResult
    GraphTokenizerMLMBatch

Compatibility Aliases
---------------------

.. py:class:: GraphSerializer

   Alias of :class:`FrequencyGuidedEulerianSerializer`; the default protocol
   is official and does not support complete graph recovery.

.. py:class:: BPEEngine

   Alias of :class:`GraphBPE`, with the same arguments and codebook contract.

See :doc:`../notes/graph_tokenizer` for protocol, backend and artifact constraints.

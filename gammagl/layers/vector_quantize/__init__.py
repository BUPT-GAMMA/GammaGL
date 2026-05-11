__all__ = ['VectorQuantize_E', 'VectorQuantize_R']


def __getattr__(name):
    if name == 'VectorQuantize_E':
        from .vq_euclidean import VectorQuantize_E
        return VectorQuantize_E
    if name == 'VectorQuantize_R':
        from .vq_riemann import VectorQuantize_R
        return VectorQuantize_R
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

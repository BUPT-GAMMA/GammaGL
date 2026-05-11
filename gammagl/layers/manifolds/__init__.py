# SciPy >= 1.11 removed scalar_search_wolfe2 from linesearch and uses
# a custom __getattr__ that raises AttributeError on access.
# geoopt imports it, so we must patch it before importing geoopt.
import sys as _sys

_scipy_ls = _sys.modules.get('scipy.optimize.linesearch')
if _scipy_ls is None:
    import scipy.optimize.linesearch as _scipy_ls

_missing = []
if not hasattr(_scipy_ls, 'scalar_search_wolfe2'):
    _missing.append('scalar_search_wolfe2')
if not hasattr(_scipy_ls, 'scalar_search_armijo'):
    _missing.append('scalar_search_armijo')

if _missing:
    try:
        from scipy.optimize._linesearch import scalar_search_wolfe2 as _ssw2
    except ImportError:
        try:
            from scipy.optimize._linesearch import scalar_search_wolfe1 as _ssw2
        except ImportError:
            _ssw2 = None
    try:
        from scipy.optimize._linesearch import scalar_search_armijo as _ssa
    except ImportError:
        _ssa = None

    if 'scalar_search_wolfe2' in _missing and _ssw2 is not None:
        _scipy_ls.__dict__['scalar_search_wolfe2'] = _ssw2
    if 'scalar_search_armijo' in _missing and _ssa is not None:
        _scipy_ls.__dict__['scalar_search_armijo'] = _ssa

    del _ssw2, _ssa
del _sys, _scipy_ls, _missing

__all__ = ['Lorentz', 'Sphere', 'Euclidean', 'ProductSpace']


def __getattr__(name):
    if name == 'Lorentz':
        from .lorentz import Lorentz
        return Lorentz
    if name == 'Sphere':
        from .sphere import Sphere
        return Sphere
    if name == 'Euclidean':
        from .euclidean import Euclidean
        return Euclidean
    if name == 'ProductSpace':
        from .product import ProductSpace
        return ProductSpace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

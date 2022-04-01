from distutils.core import setup
import numpy as np
import setuptools

from Cython.Build import cythonize

path = "gammagl/utils/sample.pyx"
setup(
    name='sample_adj',
    ext_modules=cythonize(path),
    include_dirs=[np.get_include()]
)
# msvc version must > 14.0, else cant compile
# python setup_sample_adj.py build_ext --inplace
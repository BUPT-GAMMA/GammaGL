#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# cython compile
try:
    from Cython.Build import cythonize
except ImportError:

    def cythonize(*args, **kwargs):
        """cythonize"""
        from Cython.Build import cythonize
        return cythonize(*args, **kwargs)


class CustomBuildExt(_build_ext):
    """CustomBuildExt"""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


compile_extra_args = ["-std=c++11"]
link_extra_args = []
extensions = [
    Extension(
        "gammagl.sample",
        sources=[os.path.join("gammagl", "sample.pyx")],
        language="c++",
        extra_compile_args=compile_extra_args,
        extra_link_args=link_extra_args, ),
]

install_requires = ['numpy', 'scipy', 'pytest', 'cython', 'tensorflow', 'tensorlayerx']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]

setup(
    name="gammagl",
    version="0.0.1",
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",

    cmdclass={'build_ext': CustomBuildExt},
    ext_modules=extensions,

    description=" ",

    url="https://github.com/BUPT-GAMMA/GammaGL",
    download_url="https://github.com/BUPT-GAMMA/GammaGL",

    python_requires='>=3.7',

    packages=find_packages(),

    install_requires=install_requires,
    include_package_data=True,

    classifiers=classifiers
)

# python setup.py build_ext --inplace
# python setup.py install

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext

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

    def build_extensions(self):
        self.compiler.parallel_compile = 8
        super().build_extensions()


compile_extra_args = ["-std=c++11"]


def c_extension(ext_dir, module, dep_src_list=[], **kwargs):
    if ext_dir[-1] != '/':
        ext_dir += '/'

    if len(dep_src_list) != 0 and not (dep_src_list[0].endswith('.cpp') or dep_src_list[0].endswith('.c')):
        raise NameError("Need file names, not module names!")

    for i in range(len(dep_src_list)):
        dep_src_list[i] = f"{ext_dir}{dep_src_list[i]}"

    return Pybind11Extension(f"{ext_dir}_{module}".replace("/", "."),
                             sources=[f"{ext_dir}{module}.cpp"] + dep_src_list,
                             extra_compile_args=compile_extra_args,
                             **kwargs
                             )


link_extra_args = []

extensions = [
    Extension(
        "gammagl.sample",
        sources=[os.path.join("gammagl", "sample.pyx")],
        language="c++",
        extra_compile_args=compile_extra_args,
        extra_link_args=link_extra_args, ),
    c_extension("gammagl/sparse/", "convert", ),
    c_extension("gammagl/sparse/", "neighbor_sample", dep_src_list=["utils.cpp"],
                include_dirs=["third_party/parallel_hashmap/"]),
    c_extension("gammagl/sparse/", "saint", dep_src_list=["utils.cpp"]),
    c_extension("gammagl/sparse/", "sparse", dep_src_list=["utils.cpp"]),
    c_extension("gammagl/ops/include/", "unique"),

]

install_requires = ['numpy', 'scipy', 'pytest', 'cython', 'tensorlayerx', 'rich', 'tqdm', 'pybind11']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]

setup(
    name="gammagl",
    version="0.1.0",
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
    classifiers=classifiers,
    include_package_data=True,
    package_data={
        "gammagl": ["*.json"]
    }
)

# python setup.py build_ext --inplace
# python setup.py install

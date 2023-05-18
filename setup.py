#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import platform
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


compile_extra_args = []

if platform.system() == 'Windows':
    compile_extra_args.append('/std:c++17')
else:
    compile_extra_args.append('-std=c++17')


# multi modules for development
def cpp_extension(ext_dir, module, dep_src_list=[], **kwargs):
    if len(dep_src_list) != 0 and not (dep_src_list[0].endswith('.cpp') or dep_src_list[0].endswith('.cc')):
        raise NameError("Need file names, not module names!")

    for i in range(len(dep_src_list)):
        dep_src_list[i] = osp.join(ext_dir, "cpu", dep_src_list[i])

    include_dirs = [ext_dir, "gammagl/ops"] + kwargs.get("include_dirs", [])
    if "include_dirs" in kwargs:
        del kwargs["include_dirs"]

    return Pybind11Extension(osp.join(ext_dir, f"_{module}").replace("/", "."),
                             sources=[osp.join(ext_dir, "cpu", f"{module}.cpp")] + dep_src_list,
                             include_dirs=include_dirs,
                             extra_compile_args=compile_extra_args,
                             **kwargs
                             )


link_extra_args = []

[osp.join(osp.abspath('.'), f) for f in os.listdir(osp.abspath('.'))]


# single-library
def load_extensions():
    ops_list = ["sparse", "segment", "tensor"]
    ops_third_party_deps = [['parallel_hashmap'], [], []]

    ops_root = osp.join('gammagl', 'ops')

    extensions = []
    for i in range(len(ops_list)):
        ops_prefix = ops_list[i]
        # ops_type = ["cpu","cuda"]
        ops_types = ["cpu"]
        ops_dir = osp.join(ops_root, ops_prefix)
        for ops_type in ops_types:
            src_dir = osp.join(ops_dir, ops_type)
            src_files = os.listdir(src_dir)
            extensions.append(Pybind11Extension(
                osp.join(ops_dir, f'_{ops_prefix}').replace(osp.sep, "."),
                sources=[osp.join(src_dir, f) for f in src_files],
                include_dirs=[osp.join('third_party', d) for d in ops_third_party_deps[i]],
                extra_compile_args=compile_extra_args,
            ))

    return extensions


extensions = [
    Extension(
        "gammagl.sample",
        sources=[os.path.join("gammagl", "sample.pyx")],
        language="c++",
        extra_compile_args=compile_extra_args,
        extra_link_args=link_extra_args, ),

    cpp_extension("gammagl/ops/sparse/", "convert", ),
    cpp_extension("gammagl/ops/sparse/", "sample", dep_src_list=["utils.cpp"]),
    cpp_extension("gammagl/ops/sparse/", "neighbor_sample", dep_src_list=["utils.cpp"],
                  include_dirs=["third_party/parallel_hashmap/"]),
    cpp_extension("gammagl/ops/sparse/", "saint", dep_src_list=["utils.cpp"]),
    cpp_extension("gammagl/ops/sparse/", "rw", dep_src_list=["utils.cpp"]),
    cpp_extension("gammagl/ops/tensor/", "unique"),

]

install_requires = ['numpy', 'scipy', 'pytest', 'cython', 'tensorlayerx', 'rich', 'tqdm', 'pybind11']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]

setup(
    name="gammagl",
    version="0.2.0",
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    cmdclass={'build_ext': CustomBuildExt},
    # ext_modules=extensions,
    ext_modules=load_extensions(),
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

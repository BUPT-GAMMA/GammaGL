#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
# from pybind11.setup_helpers import Pybind11Extension, build_ext
from gammagl.utils.ggl_build_extension import BuildExtension, PyCudaExtension, PyCPUExtension

# TODO will depend on different host
WITH_CUDA = False

# class CustomBuildExt(_build_ext):
#     """CustomBuildExt"""
#
#     def finalize_options(self):
#         _build_ext.finalize_options(self)
#         # Prevent numpy from thinking it is still in its setup process:
#         __builtins__.__NUMPY_SETUP__ = False
#         import numpy
#         self.include_dirs.append(numpy.get_include())
#
#     def build_extensions(self):
#         self.compiler.parallel_compile = 8
#         super().build_extensions()


compile_extra_args = []

# if platform.system() == 'Windows':
#     compile_extra_args.append('/std:c++17')
# else:
#     compile_extra_args.append('-std=c++17')

ext_root = osp.join("gammagl", "ops")


# multi cpu modules for development
def cpu_extension(ext_lib, module, dep_src_list=[], **kwargs):
    ext_dir = osp.join(ext_root, ext_lib)
    if len(dep_src_list) != 0 and not (dep_src_list[0].endswith('.cpp') or dep_src_list[0].endswith('.cc')):
        raise NameError("Not a valid source name")

    for i in range(len(dep_src_list)):
        dep_src_list[i] = osp.join(ext_dir, "cpu", dep_src_list[i])

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += [ext_root, ext_dir]
    kwargs["include_dirs"] = include_dirs

    return PyCPUExtension(osp.join(ext_dir, f"_{module}").replace(osp.sep, "."),
                          sources=[osp.join(ext_dir, "cpu", f"{module}.cpp")] + dep_src_list,
                          **kwargs
                          )


# multi cuda modules for development
def cuda_extension(ext_lib, module, dep_src_list=[], **kwargs):
    ext_dir = osp.join(ext_root, ext_lib)
    if len(dep_src_list) != 0 \
            and (not dep_src_list[0].endswith('.cu')
                 or not dep_src_list[0].endswith('.cpp')
                 or not dep_src_list[0].endswith('.cc')):
        raise NameError("Not a valid source name")

    for i in range(len(dep_src_list)):
        dep_src_list[i] = osp.join(ext_dir, "cuda", dep_src_list[i])

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += [ext_root, ext_dir]
    kwargs["include_dirs"] = include_dirs

    return PyCudaExtension(osp.join(ext_dir, f"_{module}_cuda").replace(osp.sep, "."),
                           sources=[osp.join(ext_dir, "cuda", f"{module}.cu")] + dep_src_list,
                           **kwargs
                           )


link_extra_args = []


# [osp.join(osp.abspath('.'), f) for f in os.listdir(osp.abspath('.'))]


# Start to include cuda ops, if no cuda found, will only compile cpu ops
# single-library
def load_extensions():
    ops_list = ["sparse", "segment", "tensor"]
    ops_third_party_deps = [['parallel_hashmap'], [], []]

    ops_root = osp.join('gammagl', 'ops')

    extensions = []
    for i in range(len(ops_list)):
        ops_prefix = ops_list[i]

        if WITH_CUDA:
            ops_types = ["cpu", "cuda"]
        else:
            ops_types = ["cpu"]
        ops_dir = osp.join(ops_root, ops_prefix)
        for ops_type in ops_types:
            is_cuda_ext = ops_type == "cuda"
            src_dir = osp.join(ops_dir, ops_type)
            src_files = os.listdir(src_dir)
            if not src_files:
                continue
            if not is_cuda_ext:
                extensions.append(PyCPUExtension(
                    osp.join(ops_dir, f'_{ops_prefix}').replace(osp.sep, "."),
                    sources=[osp.join(src_dir, f) for f in src_files],
                    include_dirs=[osp.join('third_party', d) for d in ops_third_party_deps[i]],
                ))
            else:
                extensions.append(PyCudaExtension(
                    osp.join(ops_dir, f'_{ops_prefix}_cuda').replace(osp.sep, "."),
                    sources=[osp.join(src_dir, f) for f in src_files],
                    include_dirs=[osp.join('third_party', d) for d in ops_third_party_deps[i]],
                ))

    return extensions


extensions = [
    # Extension(
    #     "gammagl.sample",
    #     sources=[os.path.join("gammagl", "sample.pyx")],
    #     language="c++",
    #     extra_compile_args=compile_extra_args,
    #     extra_link_args=link_extra_args, ),

    # cpp_extension("gammagl/ops/sparse/", "convert", ),
    # cpp_extension("gammagl/ops/sparse/", "sample", dep_src_list=["utils.cpp"]),
    # cpp_extension("gammagl/ops/sparse/", "neighbor_sample", dep_src_list=["utils.cpp"],
    #               include_dirs=["third_party/parallel_hashmap/"]),
    # cpp_extension("gammagl/ops/sparse/", "saint", dep_src_list=["utils.cpp"]),
    # cpp_extension("gammagl/ops/sparse/", "rw", dep_src_list=["utils.cpp"]),
    # cpp_extension("gammagl/ops/tensor/", "unique"),

    cpu_extension("sparse", "convert"),
    # cpu_extension("sparse", "sample", dep_src_list=["utils.cpp"]),
    # cpu_extension("sparse", "neighbor_sample", dep_src_list=["utils.cpp"],
    #               include_dirs=["third_party/parallel_hashmap/"]),
    # cpu_extension("sparse", "saint", dep_src_list=["utils.cpp"]),
    # cpu_extension("sparse", "rw", dep_src_list=["utils.cpp"]),
    # cpu_extension("tensor", "unique"),
    cuda_extension("sparse", "convert")

]

install_requires = ['numpy', 'scipy', 'pytest', 'tensorlayerx', 'rich', 'tqdm', 'pybind11']

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
    cmdclass={'build_ext': BuildExtension},
    ext_modules=extensions,
    # ext_modules=load_extensions(),
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

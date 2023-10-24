#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
# from pybind11.setup_helpers import Pybind11Extension, build_ext
# from gammagl.utils.ggl_build_extension import BuildExtension, PyCudaExtension, PyCPUExtension
from gammagl.utils.ggl_build_extension import PyCudaExtension, PyCPUExtension
import subprocess
import tensorlayerx as tlx
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# TODO will depend on different host
WITH_CUDA = False

ext_root = osp.join("gammagl", "ops")
cuda_macro = ('COMPILE_WITH_CUDA', None)
link_extra_args = []

def is_src_file(filename: str):
    return filename.endswith("cpp") \
           or filename.endswith("cu")

# Start to include cuda ops, if no cuda found, will only compile cpu ops
# single-library
def load_extensions():
    ops_list = ["sparse", "segment", "tensor"]
    ops_third_party_deps = [['parallel_hashmap'], [], []]  # 这里为啥会有两个空列表，直接[['par']]不行？

    mpops_list = ["torch_ext"]

    # 不直接使用gammagl/ops的原因是因为linux和win的路径所用符号不同
    # linux为/ win为\
    ops_root = osp.join('gammagl', 'ops')
    mpops_root = osp.join('gammagl', 'mpops')

    extensions = []
    file_list = []
    for i in range(len(mpops_list)):
        mpops_prefix = mpops_list[i]

        if WITH_CUDA:
            mpops_types = ["src", "cpu", "cuda"]
        else:
            mpops_types = ["src", "cpu"]
        mpops_dir = osp.join(mpops_root, mpops_prefix)
        for mpops_type in mpops_types:
            src_dir = osp.join(mpops_dir, mpops_type)
            src_files = filter(is_src_file, os.listdir(src_dir))
            if not src_files:
                continue
            file_list.extend([osp.join(mpops_type, f) for f in src_files])
    
        if not WITH_CUDA:
            extensions.append(CppExtension(
                name = osp.join(mpops_dir, f'_{mpops_prefix}').replace(osp.sep, "."),
                sources = [osp.join(mpops_dir, f) for f in file_list]
            ))
        else:
            extensions.append(CUDAExtension(
                name = osp.join(mpops_dir, f'_{mpops_prefix}').replace(osp.sep, "."),
                sources = [osp.join(mpops_dir, f) for f in file_list],
                define_macros=[
                    cuda_macro,
                    # omp_macro,
                ]
            ))
    
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
            src_files = filter(is_src_file, os.listdir(src_dir))
            if not src_files:
                continue
            if not is_cuda_ext:
                extensions.append(CppExtension(
                    name = osp.join(ops_dir, f'_{ops_prefix}').replace(osp.sep, "."),
                    sources = [osp.join(src_dir, f) for f in src_files],
                    include_dirs = [osp.join('third_party', d) for d in ops_third_party_deps[i]],
                    extra_compile_args = ['-std=c++17']
                ))
            else:
                extensions.append(CUDAExtension(
                    name = osp.join(ops_dir, f'_{ops_prefix}_cuda').replace(osp.sep, "."),
                    sources = [osp.join(src_dir, f) for f in src_files],
                    include_dirs = [osp.join('third_party', d) for d in ops_third_party_deps[i]],
                    extra_compile_args = ['-std=c++17']
                ))

    return extensions


install_requires = ['numpy<=1.23.5', 'pandas', 'numba', 'scipy', 'protobuf==3.19.6', 'pyparsing',
                    'tensorboardx<=2.5', 'pytest', 'tensorlayerx', 'rich', 'tqdm', 'pybind11', 'panda<=2.0.3']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


setup(
    name="gammagl",
    version="0.3.0",
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    cmdclass={'build_ext': BuildExtension},
    ext_modules=load_extensions(),  # 在这调用了gammagl写的extension模块，但是我们的目标是打包这个gammagl，这样子是否可行
    setup_requires=[],  # import导入了platform、os，为什么不写
    description=" ",
    long_description=readme(),
    url="https://github.com/BUPT-GAMMA/GammaGL",
    download_url="https://github.com/BUPT-GAMMA/GammaGL",
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=classifiers,
    include_package_data=True,
    # 这里是否和MANIFEST.in重复了
    # package_data={
    #     "gammagl": ["*.json"]
    # }
)

# python setup.py build_ext --inplace
# python setup.py install

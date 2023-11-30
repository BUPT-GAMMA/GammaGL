#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
# from pybind11.setup_helpers import Pybind11Extension, build_ext
from gammagl.utils.ggl_build_extension import BuildExtension, PyCudaExtension, PyCPUExtension
import subprocess

# TODO will depend on different host
WITH_CUDA = False


ext_root = osp.join("gammagl", "ops")


# multi cpu modules for development
def CPUExtension(ext_lib, module, dep_src_list=[], **kwargs):
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
def CUDAExtension(ext_lib, module, dep_src_list=[], **kwargs):
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


def is_src_file(filename:str):
    return filename.endswith("cpp") \
        or filename.endswith("cu")


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
            src_files = filter(is_src_file, os.listdir(src_dir))
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


install_requires = ['numpy<=1.23.5', 'pandas', 'numba', 'scipy', 'protobuf==3.19.6', 'pyparsing',
                    'tensorboardx<=2.5', 'pytest', 'tensorlayerx', 'rich', 'tqdm', 'pybind11', 'panda<=2.0.3']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]

setup(
    name="gammagl",
    version="0.3.0",
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    cmdclass={'build_ext': BuildExtension},
    ext_modules=load_extensions(),
    description=" ",
    url="https://github.com/BUPT-GAMMA/GammaGL",
    download_url="https://github.com/BUPT-GAMMA/GammaGL",
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=classifiers,
    include_package_data=True,
    package_data={
        "gammagl": ["*.json"]
    }
)

# os.chdir('gammagl/mpops/torch_ext')
# subprocess.call('python setup.py install', shell=True)

# python setup.py build_ext --inplace
# python setup.py install

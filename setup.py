#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
from setuptools import setup, find_packages

# Prefer torch backend during build to avoid TensorFlow-side imports/noise.
os.environ.setdefault("TL_BACKEND", "torch")

# Prefer TLX build helpers when available; fallback to torch cpp_extension.
# If neither is available (e.g. in pip build isolation), provide dummy fallbacks.
_extensions_available = False
try:
    from tensorlayerx.utils import PyCppExtension, PyCUDAExtension, PyBuildExtension
    _extensions_available = True
except Exception:
    try:
        from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

        def PyCppExtension(*args, **kwargs):
            kwargs.pop("use_torch", None)
            return CppExtension(*args, **kwargs)

        def PyCUDAExtension(*args, **kwargs):
            kwargs.pop("use_torch", None)
            return CUDAExtension(*args, **kwargs)

        PyBuildExtension = BuildExtension
        _extensions_available = True
    except Exception:
        # Neither TLX nor torch available at build time – provide dummy stubs.
        # The C++ extensions will be skipped; Python-only install will still work.
        PyCppExtension = None
        PyCUDAExtension = None

        class PyBuildExtension:
            def __new__(cls, *args, **kwargs):
                return None

# CUDA build switch:
# - default enabled
# - export WITH_CUDA=0 to force CPU-only extensions
WITH_CUDA = os.environ.get("WITH_CUDA", "1") not in ("0", "false", "False")

cuda_macro = ('COMPILE_WITH_CUDA', True)
omp_macro = ('COMPLIE_WITH_OMP', True)  # Note: OpenMP needs gcc>4.2.0
compile_args = {
    'cxx': ['-fopenmp', '-std=c++17']
}

def is_src_file(filename: str):
    return filename.endswith("cpp") \
           or filename.endswith("cu")

def load_mpops_extensions():
    mpops_list = ["torch_ext"]
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
            if not osp.exists(src_dir):
                print(f"No source files found in directory: {src_dir}")
                continue
            src_files = filter(is_src_file, os.listdir(src_dir))
            if not src_files:
                continue
            file_list.extend([osp.join(src_dir, f) for f in src_files])

        if not WITH_CUDA:
            extensions.append(PyCppExtension(
                name=osp.join(mpops_dir, f'_{mpops_prefix}').replace(osp.sep, "."),
                sources=[f for f in file_list],
                extra_compile_args=compile_args,
                use_torch=True
            ))
        else:
            extensions.append(PyCUDAExtension(
                name=osp.join(mpops_dir, f'_{mpops_prefix}').replace(osp.sep, "."),
                sources=[f for f in file_list],
                define_macros=[
                    cuda_macro,
                    omp_macro
                ],
                extra_compile_args=compile_args,
                use_torch=True
            ))

    return extensions


def load_ops_extensions():
    ops_list = ["sparse", "segment", "tensor"]
    ops_third_party_deps = [['parallel_hashmap'], [], []]

    ops_root = osp.join('gammagl', 'ops')

    extensions = []

    for i in range(len(ops_list)):
        ops_prefix = ops_list[i]

        # ops_types = ["cpu"]
        if WITH_CUDA:
            ops_types = ["cpu", "cuda"]
        else:
            ops_types = ["cpu"]
        ops_dir = osp.join(ops_root, ops_prefix)
        for ops_type in ops_types:
            is_cuda_ext = ops_type == "cuda"
            src_dir = osp.join(ops_dir, ops_type)
            if not osp.exists(src_dir):
                print(f"No source files found in directory: {src_dir}")
                continue
            src_files = filter(is_src_file, os.listdir(src_dir))
            if not src_files:
                continue
            if not is_cuda_ext:
                extensions.append(PyCppExtension(
                    name=osp.join(ops_dir, f'_{ops_prefix}').replace(osp.sep, "."),
                    sources=[osp.join(src_dir, f) for f in src_files],
                    include_dirs=[osp.abspath(osp.join('third_party', d)) for d in ops_third_party_deps[i]],
                    extra_compile_args=['-std=c++17']
                ))
            else:
                extensions.append(PyCUDAExtension(
                    name=osp.join(ops_dir, f'_{ops_prefix}_cuda').replace(osp.sep, "."),
                    sources=[osp.join(src_dir, f) for f in src_files],
                    include_dirs=[osp.abspath(osp.join('third_party', d)) for d in ops_third_party_deps[i]],
                    extra_compile_args=['-std=c++17'],
                    use_torch=True
                ))

    return extensions


# Build switch for optional GammaGL ops extensions (sparse/segment/tensor).
# These require third_party dependencies (e.g. parallel_hashmap submodule).
BUILD_GGL_OPS = os.environ.get("BUILD_GGL_OPS", "0") in ("1", "true", "True")

# Start to include cuda ops, if no cuda found, will only compile cpu ops
def load_extensions():
    if not _extensions_available:
        print("C++ build helpers not available (torch/tensorlayerx missing in build env).")
        print("Skipping C++ extensions – Python-only install will proceed.")
        return []
    extensions = load_mpops_extensions()
    if BUILD_GGL_OPS:
        extensions += load_ops_extensions()
    else:
        print("Skip building gammagl/ops extensions (set BUILD_GGL_OPS=1 to enable).")

    return extensions

install_requires = ['numpy==1.24', 'pandas', 'numba==0.59.0', 'scipy', 'protobuf', 'pyparsing', 'rdkit',
                    'tensorboardx', 'pytest', 'tensorlayerx', 'rich', 'tqdm', 'pybind11', 'panda', 'ninja==1.11.1.1']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
]


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


_ext_modules = load_extensions()
_cmdclass = {'build_ext': PyBuildExtension} if _extensions_available else {}

setup(
    name="gammagl",
    version="0.6.0",
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    cmdclass=_cmdclass,
    ext_modules=_ext_modules,
    description=" ",
    long_description=readme(),
    url="https://github.com/BUPT-GAMMA/GammaGL",
    download_url="https://github.com/BUPT-GAMMA/GammaGL",
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=classifiers,
    include_package_data=True
)

# python setup.py build_ext --inplace
# python setup.py install

# clang-format -style=file -i ***.cpp
# find ./ -type f \( -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.c' -or -name '*.cc' -or -name '*.cu' \) -print | xargs clang-format -style=file -i

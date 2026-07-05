#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import shutil
from setuptools import setup, find_packages
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
# from ggl_build_extension import PyCudaExtension, PyCPUExtension
try:
    from tensorlayerx.utils import PyCppExtension, PyCUDAExtension, PyBuildExtension
except ImportError as exc:
    raise RuntimeError(
        "GammaGL source builds require PyTorch and the GAMMA Lab TensorLayerX "
        "branch first. Install a backend, TensorLayerX, and build prerequisites: "
        "pip install git+https://github.com/dddg617/tensorlayerx.git@nightly "
        "pybind11 ninja"
    ) from exc

VERSION = "0.6.0"
TLX_NIGHTLY = "tensorlayerx @ git+https://github.com/dddg617/tensorlayerx.git@nightly"


def _has_cuda_toolkit():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        nvcc_path = osp.join(cuda_home, "bin", "nvcc")
        cuda_header = osp.join(cuda_home, "include", "cuda.h")
        if osp.exists(nvcc_path) or osp.exists(cuda_header):
            return True
    return shutil.which("nvcc") is not None


def _resolve_with_cuda():
    value = os.environ.get("GAMMAGL_WITH_CUDA", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        if not _has_cuda_toolkit():
            print("GAMMAGL_WITH_CUDA=1 was set, but nvcc/CUDA headers were not detected.")
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value == "auto":
        return _has_cuda_toolkit()
    raise ValueError("GAMMAGL_WITH_CUDA must be one of 0, 1, or auto")


WITH_CUDA = _resolve_with_cuda()
print(f"GammaGL extension build: GAMMAGL_WITH_CUDA={'1' if WITH_CUDA else '0'}")

cuda_macro = ('COMPILE_WITH_CUDA', True)
omp_macro = ('COMPLIE_WITH_OMP', True)  # Note: OpenMP needs gcc>4.2.0
compile_args = {
    'cxx': ['-fopenmp', '-std=c++17']
}

def is_src_file(filename: str):
    return filename.endswith((".cpp", ".cu"))

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


# Start to include cuda ops, if no cuda found, will only compile cpu ops
def load_extensions():
    extensions = load_mpops_extensions() + load_ops_extensions()

    return extensions

install_requires = [
    'numpy>=1.24,<2.0',
    'pandas',
    'numba>=0.59.0',
    'scipy',
    'protobuf',
    'pyparsing',
    'tensorboardX',
    'rich',
    'tqdm',
    TLX_NIGHTLY,
]

build_requires = [
    'pybind11>=2.10',
    'ninja>=1.11',
]

llm_gfm_requires = [
    'torch>=2.1',
    'transformers>=4.31',
    'sentence-transformers',
    'huggingface-hub',
    'accelerate',
    'peft',
    'openai>=1.0',
    'torch-geometric',
]

extras_require = {
    'build': build_requires,
    'dev': build_requires + ['pytest', 'ruff'],
    'docs': [
        'sphinx',
        'sphinx-rtd-theme',
        'sphinx-markdown-tables',
        'sphinx-intl',
        'recommonmark',
        'sphinx-copybutton==0.4.0',
        'nbsphinx',
    ],
    'defog': ['rdkit', 'networkx'],
    'llm': [
        'torch>=2.1',
        'transformers>=4.31',
        'sentence-transformers',
        'huggingface-hub',
        'accelerate',
        'peft',
    ],
    'gfm': llm_gfm_requires,
    'llm-gfm': llm_gfm_requires,
}

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
    version=VERSION,
    author="BUPT-GAMMA LAB",
    author_email="tyzhao@bupt.edu.cn",
    maintainer="Tianyu Zhao",
    license="Apache-2.0 License",
    cmdclass={'build_ext': PyBuildExtension},
    ext_modules=load_extensions(),
    description=" ",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/BUPT-GAMMA/GammaGL",
    download_url="https://github.com/BUPT-GAMMA/GammaGL",
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    include_package_data=True
)

# clang-format -style=file -i ***.cpp
# find ./ -type f \( -name '*.h' -or -name '*.hpp' -or -name '*.cpp' -or -name '*.c' -or -name '*.cc' -or -name '*.cu' \) -print | xargs clang-format -style=file -i

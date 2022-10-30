from numpy import source
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def get_exts():
    if torch.cuda.is_available():
        return [
            CUDAExtension(
                name='torch_segment', # Note: same with TORCH_LIBRARY (import)
                sources=['segment_max.cpp', 'segment_max_cuda.cu'],
                define_macros=[
                    ('COMPLIE_WITH_CUDA', None),
                    ('COMPILE_WITH_OMP', None) # Note: OpenMP needs gcc>4.2.0
                ]
            )
        ]
    else:
        return [
            CppExtension(
                name='torch_segment', 
                sources=['segment_max.cpp'],
                define_macros=[
                    ('COMPILE_WITH_OMP', None)
                ]
            )
        ]


setup(
    name='torch_segment', # PyPI name (install)
    ext_modules=get_exts(),
    cmdclass={'build_ext': BuildExtension}
)

from numpy import source
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def get_exts():
    if torch.cuda.is_available():
        return [
            CUDAExtension(
                name='torch_segment', # Note: same with TORCH_LIBRARY (import)
                sources=['segment_max.cpp'],
                define_macros=[
                    ('COMPLIE_WITH_CUDA', None),
                    ('COMPILE_WITH_OMP', None) # Note: OpenMP needs gcc>4.2.0
                ],
                extra_compile_args={
                    'cxx':['-fopenmp']
                }
            )
        ]
    else:
        return [
            CppExtension(
                name='torch_segment', 
                sources=[
                    'segment_max.cpp',
                    'cpu/segment_max_cpu.cpp'
                    ],
                define_macros=[
                    ('COMPILE_WITH_OMP', None)
                ],
                extra_compile_args={
                    'cxx':['-fopenmp']
                }
            ),
            CppExtension(
                name='torch_gspmm', 
                sources=[
                    'gspmm.cpp',
                    'cpu/spmm_sum_cpu.cpp'
                    ],
                define_macros=[
                    ('COMPILE_WITH_OMP', None)
                ],
                extra_compile_args={
                    'cxx':['-fopenmp']
                }
            )
        ]


setup(
    name='torch_ext', # PyPI name (install)
    ext_modules=get_exts(),
    cmdclass={'build_ext': BuildExtension}
)

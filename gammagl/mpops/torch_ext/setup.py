from numpy import source
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

cuda_macro = ('COMPILE_WITH_CUDA', None)
# omp_macro = ('COMPILE_WITH_OMP', None) # Note: OpenMP needs gcc>4.2.0
# compile_args = {
#     'cxx':['-fopenmp']
# }

def get_exts():
    if torch.cuda.is_available():
        return [
            CUDAExtension(
                name='torch_segment', # Note: same with TORCH_LIBRARY (import)
                sources=[
                    'segment_max.cpp',
                    'cpu/segment_max_cpu.cpp',
                    'cuda/segment_max_cuda.cu'
                    ],
                define_macros=[
                    cuda_macro,
                    # omp_macro,
                    ],
                # extra_compile_args=compile_args
            ),
            CUDAExtension(
                name='torch_gspmm', # Note: same with TORCH_LIBRARY (import)
                sources=[
                    'gspmm.cpp',
                    'cpu/spmm_sum_cpu.cpp',
                    'cuda/spmm_sum_cuda.cu'
                    ],
                define_macros=[
                    cuda_macro, 
                    # omp_macro,
                    ],
                # extra_compile_args=compile_args
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
                # define_macros=[omp_macro],
                # extra_compile_args=compile_args
            ),
            CppExtension(
                name='torch_gspmm', 
                sources=[
                    'gspmm.cpp',
                    'cpu/spmm_sum_cpu.cpp'
                    ],
                # define_macros=[omp_macro],
                # extra_compile_args=compile_args
            )
        ]


setup(
    name='torch_ext', # PyPI name (install)
    ext_modules=get_exts(),
    cmdclass={'build_ext': BuildExtension}
)

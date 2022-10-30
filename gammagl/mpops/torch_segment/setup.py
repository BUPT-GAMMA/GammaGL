import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def get_exts():
    if torch.cuda.is_available():
        return [
            CUDAExtension(
                'torch_segment', 
                ['segment_max.cpp', 'segment_max_cuda.cu'],
                define_macros=[('COMPLIE_WITH_CUDA', None)]
                )]
    else:
        return [
            CppExtension(
                'torch_segment', 
                ['segment_max.cpp']
                )]


setup(
    name='torch_segment',
    ext_modules=get_exts(),
    cmdclass={'build_ext': BuildExtension}
)

from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

# Provide the missing classes
PyCppExtension = CppExtension
PyCUDAExtension = CUDAExtension
PyBuildExtension = BuildExtension

# Also provide other utilities that might be needed
__all__ = ['PyCppExtension', 'PyCUDAExtension', 'PyBuildExtension']

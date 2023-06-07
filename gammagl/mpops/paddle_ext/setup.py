import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

cuda_macro = ('COMPILE_WITH_CUDA', None) # Paddle offer `PADDLE_WITH_CUDA` macro
# omp_macro = ('COMPILE_WITH_OMP', None) # Note: OpenMP needs gcc>4.2.0
# compile_args = {
#     'cxx':['-fopenmp']
# }

def get_exts():
    if paddle.is_compiled_with_cuda():
        return CUDAExtension(
            # name="paddle_segment", # paddle not support, and the name will get from setup
            sources=[
                'segment_sum.cpp',
                'cpu/segment_sum_cpu.cpp',
                'cuda/segment_sum_cuda.cu',
            ],
            define_macros=[
                cuda_macro, 
                # omp_macro,
                ],
            # extra_compile_args=compile_args
        )
    else:
        return CppExtension(
            sources=[
                'segment_sum.cpp',
                'cpu/segment_sum_cpu.cpp',
            ],
            # define_macros=[omp_macro],
            # extra_compile_args=compile_args
        )

setup(
    name='paddle_ext',
    ext_modules=get_exts()
)
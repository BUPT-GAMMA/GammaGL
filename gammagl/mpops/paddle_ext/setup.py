import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

def get_exts():
    if paddle.is_compiled_with_cuda():
        return CUDAExtension(
            sources=['segment_sum.cpp'],
            define_macros=[
                ('COMPILE_WITH_OMP', None),
                # ('COMPLIE_WITH_CUDA', None),  # CUDAExtension will define PADDLE_WITH_CUDA macro
            ]
        )
    else:
        return CppExtension(
            sources=['segment_sum.cpp'],
            define_macros=[
                ('COMPILE_WITH_OMP', None)
            ]
        )

setup(
        name='paddle_segment',
        ext_modules=get_exts()
    )
import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

if paddle.is_compiled_with_cuda:
    print("compile with cuda")
    setup(
        name='paddle_segment',
        ext_modules=CUDAExtension(
            sources=['segment_sum.cpp', 'segment_sum.cu']
        )
    )

else:
    print("compile with cpp")
    setup(
        name='paddle_segment',
        ext_modules=CppExtension(
            sources=['segment_sum.cpp']
        )
    )
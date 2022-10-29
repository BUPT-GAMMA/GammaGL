from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='paddle_segment',
    ext_modules=CppExtension(
        sources=['segment_sum.cpp']
    )
)

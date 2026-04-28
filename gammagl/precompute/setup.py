import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

from distutils.core import setup, Extension
from Cython.Build import cythonize
import eigency

setup(
    author='nyLiao',
    version='0.0.1',
    install_requires=['Cython>=0.2.15','eigency>=1.77'],
    packages=['little_try'],
    python_requires='>=3',
    ext_modules=cythonize(Extension(
        name='prop',
        sources=['prop.pyx'],
        language='c++',
        extra_compile_args=[
            "-std=c++14",  # 升级C++14，彻底解决模板冲突
            "-O3", 
            "-fopenmp",
            # 🔥 核心修复：解决Eigen isfinite函数冲突
            "-DEIGEN_DONT_ALIGN",
            "-DEIGEN_NO_CXX11_NUMERIC_LIMITS",
            "-DNDEBUG"
        ],
        include_dirs=[".",] + eigency.get_includes()[:2] + ["/uer/ycn/miniconda3/envs/self/include/eigen3"],
    ))
)

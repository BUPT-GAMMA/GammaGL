An cpp extension of unsorted_segment_max and GSpMM operators.   
This NAIVE implementation achieves over 10 times acceleration on multi-core CPU.

Compile Steps:  (setuptools, recommended)  
python setup.py install

Compile Steps:  (CMake, may not work)  
install and config cmake make libtorch  
mkdir build && cd build && cmake .. && make -j  

cpu dir: cpu version operators  
cuda dir: cuda version operators  
segment_{op}.cpp: operators that dispatch device  

Testing script: GammaGL/profiler/mpops/torch_ext.py


HeteroLinear Module Base on Grouped GEMM
reference  
[https://zhuanlan.zhihu.com/p/484319691](https://zhuanlan.zhihu.com/p/484319691)
[https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)  
[https://github.com/NVIDIA/cutlass/tree/master/examples/24_gemm_grouped](https://github.com/NVIDIA/cutlass/tree/master/examples/24_gemm_grouped)  
[https://github.com/pyg-team/pyg-lib](https://github.com/pyg-team/pyg-lib)

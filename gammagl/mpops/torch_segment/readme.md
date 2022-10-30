An cpp extension of unsorted_segment_max operator. This NAIVE implementation achieves over 10 times acceleration.

Compile Steps:  (setuptools, recommended)
python setup.py install

Compile Steps:  (CMake, may not work)
install and config cmake make libtorch  
mkdir build && cd build && cmake .. && make -j  

cpu dir: cpu version operators  
cuda dir: cuda version operators  
segment_{op}.cpp: operators that dispatch device  

Testing script: GammaGL/profiler/mpops/torch_ext.py
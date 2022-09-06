An cpu-only cpp extension of unsorted_segment_max operator. This NAIVE implementation achieves over 10x acceleration on a single core over the original implementation on dual-core.

Compile Steps:  
install and config cmake make libtorch  
mkdir build && cd build && cmake .. && make -j  

Testing script: GammaGL/profiler/mpops/torch_ext.py
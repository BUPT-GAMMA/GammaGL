Compile Steps:  (setuptools, recommended)  
python setup.py install

Compile Steps:  (CMake, not work)  
TODO: support cmake

> In `paddle/utils/cpp_extension/extension_utils.py:341L`, flags `"-ccbin"` & `"cc"` may cause error, since Paddle needs `nvcc` compiling with higher c++ standard while these flags may cause flag like `-std=c++14` does not work, when you are using a lower version gcc. Besides, we usually recommand to set gcc path to `CC` rather than `cc` in Linux, it will also cause error. Just annotate them.

> Please keep the version of nvcc and the paddle-cuda consist, it may occur `the provided PTX was compiled with an unsupported toolchain.`

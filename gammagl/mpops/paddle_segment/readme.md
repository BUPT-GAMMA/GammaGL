Compile Steps:  (setuptools, recommended)  
python setup.py install

Compile Steps:  (CMake, not work)  
TODO: support cmake

In `paddle/utils/cpp_extension/extension_utils.py:341L`, `"-ccbin"` `"cc"` may cause error, just annotate them.

# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/22
import copy
import glob
import os
import os.path as osp
import re
import subprocess
import sys
from typing import Optional, List

import setuptools
from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext

IS_WINDOWS = sys.platform == 'win32'
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()

COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr'
]

COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/EHsc']


def _is_cuda_file(path: str) -> bool:
    valid_ext = ['.cu', '.cuh']
    return os.path.splitext(path)[1] in valid_ext


def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None


    return cuda_home


CUDA_HOME = _find_cuda_home()


# CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')


def _join_cuda_home(*paths) -> str:
    r'''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.

    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def include_paths(cuda: bool = False) -> List[str]:
    '''
    Get the include paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific include paths.

    Returns:
        A list of include path strings.
    '''
    paths = []
    if cuda:
        cuda_home_include = _join_cuda_home('include')
        # if we have the Debian/Ubuntu packages for cuda, we get /usr as cuda home.
        # but gcc doesn't like having /usr/include passed explicitly
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
    return paths


def library_paths(cuda: bool = False) -> List[str]:
    r'''
    Get the library paths required to build a C++ or CUDA extension.

    Args:
        cuda: If `True`, includes CUDA-specific library paths.

    Returns:
        A list of library path strings.
    '''

    paths = []

    if cuda:
        if IS_WINDOWS:
            lib_dir = 'lib/x64'
        else:
            lib_dir = 'lib64'
            if (not os.path.exists(_join_cuda_home(lib_dir)) and
                    os.path.exists(_join_cuda_home('lib'))):
                # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
                # Note that it's also possible both don't exist (see
                # _find_cuda_home) - in that case we stay with 'lib64'.
                lib_dir = 'lib'

            paths.append(_join_cuda_home(lib_dir))
            # if CUDNN_HOME is not None:
            #     paths.append(os.path.join(CUDNN_HOME, lib_dir))
    return paths


class BuildExtension(build_ext, object):
    def build_extensions(self) -> None:
        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'nvcc' when
            # extra_compile_args is a dict.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

        # Register .cu, .cuh and .hip as valid source extensions.

        self.compiler.src_extensions += ['.cu', '.cuh']
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile


        def unix_cuda_flags(cflags):
            cflags = (COMMON_NVCC_FLAGS +
                      ['--compiler-options', "'-fPIC'"] +
                      cflags
                      # + _get_cuda_arch_flags(cflags)
                      )

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if (
                    _ccbin is not None
                    and not any([flag.startswith('-ccbin') or flag.startswith('--compiler-bindir') for flag in cflags])
            ):
                cflags.extend(['-ccbin', _ccbin])

            return cflags

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' or IS_WINDOWS else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [_join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags = unix_cuda_flags(cflags)

                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def win_wrap_single_compile(sources,
                                    output_dir=None,
                                    macros=None,
                                    include_dirs=None,
                                    debug=0,
                                    extra_preargs=None,
                                    extra_postargs=None,
                                    depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            append_std17_if_no_std_present(self.cflags)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn


        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_wrap_single_compile
        else:
            self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)


def PyCudaExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)

    libraries = kwargs.get('libraries', [])
    libraries.append('cudart')

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths(cuda=True)

    include_pybind11 = kwargs.pop("include_pybind11", True)
    if include_pybind11:
        # If using setup_requires, this fails the first time - that's okay
        try:
            import pybind11
            pyinc = pybind11.get_include()
            if pyinc not in include_dirs:
                include_dirs.append(pyinc)
        except ModuleNotFoundError:
            pass

    kwargs['library_dirs'] = library_dirs
    kwargs['libraries'] = libraries
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    define_macros = kwargs.get("define_macros", [])
    define_macros.append(('WITH_CUDA', None))
    kwargs["define_macros"] = define_macros

    return setuptools.Extension(name, sources, *args, **kwargs)


def PyCPUExtension(name, sources, *args, **kwargs):
    compile_extra_args = kwargs.get("compile_extra_args", [])
    if IS_WINDOWS:
        if not any(arg.startswith('/std:') for arg in compile_extra_args):
            compile_extra_args.append('/std:c++17')
    else:
        compile_extra_args.append('-std=c++17')
    kwargs["compile_extra_args"] = compile_extra_args

    return Pybind11Extension(name, sources, *args, **kwargs)

# def _get_cuda_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
#     r'''
#     Determine CUDA arch flags to use.
#
#     For an arch, say "6.1", the added compile flag will be
#     ``-gencode=arch=compute_61,code=sm_61``.
#     For an added "+PTX", an additional
#     ``-gencode=arch=compute_xx,code=compute_xx`` is added.
#
#     See select_compute_arch.cmake for corresponding named and supported arches
#     when building with CMake.
#     '''
#     # If cflags is given, there may already be user-provided arch flags in it
#     # (from `extra_compile_args`)
#     if cflags is not None:
#         for flag in cflags:
#             if 'arch' in flag:
#                 return []
#
#     # Note: keep combined names ("arch1+arch2") above single names, otherwise
#     # string replacement may not do the right thing
#     named_arches = collections.OrderedDict([
#         ('Kepler+Tesla', '3.7'),
#         ('Kepler', '3.5+PTX'),
#         ('Maxwell+Tegra', '5.3'),
#         ('Maxwell', '5.0;5.2+PTX'),
#         ('Pascal', '6.0;6.1+PTX'),
#         ('Volta', '7.0+PTX'),
#         ('Turing', '7.5+PTX'),
#         ('Ampere', '8.0;8.6+PTX'),
#     ])
#
#     supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2',
#                         '7.0', '7.2', '7.5', '8.0', '8.6']
#     valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]
#
#     # The default is sm_30 for CUDA 9.x and 10.x
#     # First check for an env var (same as used by the main setup.py)
#     # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
#     # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
#     _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
#
#     # If not given, determine what's best for the GPU / CUDA version that can be found
#     if not _arch_list:
#         arch_list = []
#         # the assumption is that the extension should run on any of the currently visible cards,
#         # which could be of different types - therefore all archs for visible cards should be included
#         for i in range(torch.cuda.device_count()):
#             capability = torch.cuda.get_device_capability(i)
#             supported_sm = [int(arch.split('_')[1])
#                             for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
#             max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
#             # Capability of the device may be higher than what's supported by the user's
#             # NVCC, causing compilation error. User's NVCC is expected to match the one
#             # used to build pytorch, so we use the maximum supported capability of pytorch
#             # to clamp the capability.
#             capability = min(max_supported_sm, capability)
#             arch = f'{capability[0]}.{capability[1]}'
#             if arch not in arch_list:
#                 arch_list.append(arch)
#         arch_list = sorted(arch_list)
#         arch_list[-1] += '+PTX'
#     else:
#         # Deal with lists that are ' ' separated (only deal with ';' after)
#         _arch_list = _arch_list.replace(' ', ';')
#         # Expand named arches
#         for named_arch, archval in named_arches.items():
#             _arch_list = _arch_list.replace(named_arch, archval)
#
#         arch_list = _arch_list.split(';')
#
#     flags = []
#     for arch in arch_list:
#         if arch not in valid_arch_strings:
#             raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
#         else:
#             num = arch[0] + arch[2]
#             flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
#             if arch.endswith('+PTX'):
#                 flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')
#
#     return sorted(list(set(flags)))

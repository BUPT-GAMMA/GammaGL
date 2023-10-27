# cmake_minimum_required(VERSION 3.21)
# project(torch_ext)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})

# find_package(Torch REQUIRED)

# # 用OpenMP的话，在计算sum、mean的时候会出现结果不稳定，每次运行的结果都不同
# # find_package(OpenMP REQUIRED)
# # if(OPENMP_FOUND)
# #     message("Found OpenMP")
# #     add_definitions(-DCOMPILE_WITH_OMP)
# #     set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
# #     set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
# # endif()

# find_package(CUDA)
# if(${CUDA_FOUND})
#     message("Found CUDA")
#     add_definitions(-DCOMPILE_WITH_CUDA)
# endif()

# set(SRC_LIST
#     src/utils.cpp
#     src/segment_mean.cpp
#     src/segment_max.cpp
#     src/segment_sum.cpp
#     )
# set(CPU
#     cpu/segment_mean_cpu.cpp
#     cpu/segment_max_cpu.cpp
#     cpu/segment_sum_cpu.cpp
#     )
# set(INCLUDE
#     include/utils.h
#     include/segment_mean.h
#     include/segment_max.h
#     include/segment_sum.h
#     cpu/segment_mean_cpu.h
#     cpu/segment_max_cpu.h
#     cpu/segment_sum_cpu.h
#     )

# include_directories(INCLUDE)

# # add_library(torch_operator SHARED ${INCLUDE} ${SRC} ${CPU})
# add_executable(torch_operator main.cpp ${SRC_LIST} ${CPU})
# target_link_libraries(torch_operator ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
# ！！！！上面为编译cpu下的算子所用cmake内容，可以直接使用


cmake_minimum_required(VERSION 3.21)
project(torch_ext)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# find_package(OpenMP REQUIRED)
# if(OPENMP_FOUND)
#     message("Found OpenMP")
#     add_definitions(-DCOMPILE_WITH_OMP)
#     set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
#     set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
# endif()

# find_package(CUDA)
# if(${CUDA_FOUND})
#     message("Found CUDA")
#     add_definitions(-DCOMPILE_WITH_CUDA)
# endif()

set(SRC_LIST
    src/utils.cpp
    src/segment_max.cpp
    src/segment_sum.cpp
    src/segment_mean.cpp
    )
set(CPU
    cpu/segment_max_cpu.cpp
    cpu/segment_sum_cpu.cpp
    cpu/segment_mean_cpu.cpp
    )
set(CUDA
    cuda/segment_max_cuda.cu
    cuda/segment_sum_cuda.cu
    cuda/segment_mean_cuda.cu
    )
set(INCLUDE
    include/utils.h
    include/segment_max.h
    include/segment_sum.h
    include/segment_mean.h
    cpu/segment_max_cpu.h
    cpu/segment_sum_cpu.h
    cpu/segment_mean_cpu.h
    cuda/segment_max_cuda.h
    cuda/segment_sum_cuda.h
    cuda/segment_mean_cuda.h
    )

include_directories(INCLUDE)

# add_library(torch_operator SHARED main.cpp ${INCLUDE} ${SRC} ${CPU})
# add_library(torch_operator SHARED ${INCLUDE} ${SRC} ${CPU})
cuda_add_executable(torch_operator main.cpp ${SRC_LIST} ${CPU} ${CUDA})
target_link_libraries(torch_operator ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})




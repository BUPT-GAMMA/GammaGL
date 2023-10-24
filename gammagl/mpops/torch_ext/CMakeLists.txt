# cmake_minimum_required(VERSION 3.21)
# project(torch_operator)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(PythonLibs REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})

# find_package(Torch REQUIRED)

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

# add_library(torch_operator SHARED segment_max.cpp segment_max_cuda.cu)
# target_link_libraries(torch_operator ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})


cmake_minimum_required(VERSION 3.21)
project(_torch_ext)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)

# 用OpenMP的话，在计算sum、mean的时候会出现结果不稳定，每次运行的结果都不同
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
    src/segment_mean.cpp
    src/segment_max.cpp
    src/segment_sum.cpp
    )
set(CPU
    cpu/segment_mean_cpu.cpp
    cpu/segment_max_cpu.cpp
    cpu/segment_sum_cpu.cpp
    )
set(INCLUDE
    include/utils.h
    include/segment_mean.h
    include/segment_max.h
    include/segment_sum.h
    cpu/segment_mean_cpu.h
    cpu/segment_max_cpu.h
    cpu/segment_sum_cpu.h
    )

include_directories(INCLUDE)

add_library(_torch_ext SHARED ${SRC_LIST} ${CPU})
# add_executable(torch_operator main.cpp ${SRC_LIST} ${CPU})
target_link_libraries(_torch_ext ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
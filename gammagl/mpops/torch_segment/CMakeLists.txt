cmake_minimum_required(VERSION 3.21)
project(torch_segment)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Torch REQUIRED)

# find_package(OpenMP REQUIRED)
# if(OPENMP_FOUND)
#     message("Found OpenMP")
#     set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
#     set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
#     set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
# endif()

add_library(torch_segment SHARED segment_max.cpp)
target_compile_features(torch_segment PRIVATE cxx_std_14)
target_link_libraries(torch_segment ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
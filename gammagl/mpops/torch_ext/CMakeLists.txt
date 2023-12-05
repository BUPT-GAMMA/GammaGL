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

file(GLOB_RECURSE SRC_LIST "src/*.cpp")
list(REMOVE_ITEM SRC_LIST "${CMAKE_SOURCE_DIR}/src/operators.cpp")
file(GLOB_RECURSE CPU "cpu/*.cpp")
file(GLOB_RECURSE CUDA "cuda/*.cu")
file(GLOB_RECURSE INCLUDE "./*.h")

message(STATUS ${SRC_LIST})
message(STATUS ${CPU})
message(STATUS ${CUDA})
message(STATUS ${INCLUDE})

include_directories(INCLUDE)

# add_library(torch_operator SHARED main.cpp ${INCLUDE} ${SRC} ${CPU})
# add_library(torch_operator SHARED ${INCLUDE} ${SRC} ${CPU})
cuda_add_executable(torch_operator main.cpp ${SRC_LIST} ${CPU} ${CUDA})
target_link_libraries(torch_operator ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

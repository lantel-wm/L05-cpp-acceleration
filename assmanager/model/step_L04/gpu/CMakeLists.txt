cmake_minimum_required(VERSION 3.5)
find_package(CUDA)
find_package(PythonLibs 3 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(gpu SHARED
  gpu.cpp
  gpu.cu)

target_link_libraries(gpu
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(gpu PROPERTIES PREFIX "")
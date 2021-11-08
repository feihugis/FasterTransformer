cmake_minimum_required(VERSION 3.8)

add_subdirectory(passes)

add_definitions(-DTORCH_CUDA=1)

set(LIB_NAME "fastseq_extension")
pybind11_add_module(${LIB_NAME} SHARED py_bind.cc)
target_link_libraries(${LIB_NAME} PUBLIC "${TORCH_LIBRARIES}" fastseq_jit_passes)

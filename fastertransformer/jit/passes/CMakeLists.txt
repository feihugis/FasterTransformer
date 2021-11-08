cmake_minimum_required(VERSION 3.8)

set(fastseq_jit_passes
  fuse_layer_norm.cc
)

add_definitions(-DTORCH_CUDA=1)


set(LIB_NAME "fastseq_jit_passes")
add_library(${LIB_NAME} SHARED ${fastseq_jit_passes})
set_target_properties(${LIB_NAME} PROPERTIES
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}" pyt_fastertransformer)
#include <torch/script.h>
#include <torch/custom_class.h>
#include "torch/csrc/cuda/Stream.h"

#include "fastertransformer/cuda/cuda_kernels.h"

namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

Tensor topK(Tensor input, int64_t k);

Tensor topK_v2(Tensor input, int64_t k);

}
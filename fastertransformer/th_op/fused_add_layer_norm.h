#include <torch/script.h>
#include <torch/custom_class.h>
#include "fastertransformer/cuda/cuda_kernels.h"

namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

// output = layer_norm(input_tensor + residual_tensor + bias, weight, bias, eps, dim)
// math for layer_norm = y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
Tensor fused_add_layer_norm(
  Tensor input_tensor, Tensor residual_tensor, Tensor linear_weight,
  Tensor linear_bias, Tensor gamma, Tensor beta);

} // namespace torch_ext
#include "fastertransformer/th_op/fused_add_layer_norm.h"
#include "fastertransformer/cuda/transformer_kernels.cuh"
#include "fastertransformer/utils/common.h"
#include "ATen/cuda/CUDAContext.h"


namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

void debug_tesnor(const Tensor& t, const std::string& name) {
  std::cout << name << " tensor : ";
  std::cout << " size: " << t.sizes();
  std::cout << " device: " << t.device();
  std::cout << " dtype: " << t.scalar_type();
  std::cout << " is_contiguous: " << t.is_contiguous() << std::endl;
}

Tensor fused_add_layer_norm(
  Tensor input_tensor, Tensor residual_tensor, Tensor linear_weight,
  Tensor linear_bias, Tensor gamma, Tensor beta){
  
  // std::cout << "++++++++++++ Run fused_add_layer_norm kernel \n" << std::endl;

  std::vector<int64_t> input_shape  = input_tensor.sizes().vec();
  int k = input_tensor.size(-1);
  int m = input_tensor.numel() / k;
  // linear_weight in pytorch needs to be tranposed
  int n = linear_weight.size(0);
 

  std::vector<int64_t> output_shape = input_shape;
  output_shape[output_shape.size() - 1] = linear_weight.size(0);
  Tensor output = at::empty({output_shape}, input_tensor.options());
  Tensor weighted_input = at::empty(output_shape, input_tensor.options());

  // debug_tesnor(input_tensor, "input_tensor");
  // debug_tesnor(residual_tensor, "residual_tensor");
  // debug_tesnor(linear_weight, "linear_weight");
  // debug_tesnor(linear_bias, "linear_bias");
  // debug_tesnor(output, "output");
  // debug_tesnor(gamma, "gamma");
  // debug_tesnor(beta, "beta");
  // debug_tesnor(weighted_input, "weighted_input");

  

  float* input_tensor_ptr = input_tensor.data_ptr<float>();
  float* weighted_input_ptr = weighted_input.data_ptr<float>();
  float* linear_weight_ptr = linear_weight.data_ptr<float>();
  float* linear_bias_ptr = linear_bias.data_ptr<float>();

  // print_to_screen<float>(linear_weight_ptr, 3, "linear_weight", linear_weight.size(-1), 8);
  // print_to_screen<float>(input_tensor_ptr, 3, "input_tensor", input_tensor.size(-1), 8);
  // print_to_screen<float>(weighted_input_ptr, 3, "weighted_input", weighted_input.size(-1), 8);

  float linear_alpha = 1.0f;
  float linear_beta = 0.0f;

  auto stream = at::cuda::getCurrentCUDAStream();

  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  
  int gemm_algo_id = 113;
  // std::cout << "------ Enter cublasGemmEx \n" << std::endl;

  // printf("m = %ld, n = %ld, k = %ld \n", linear_weight.size(0), input_tensor.size(0), linear_weight.size(-1));

  cublasStatus_t status = cublasGemmEx(cublas_handle, 
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       linear_weight.size(0),
                                       input_tensor.size(0),
                                       linear_weight.size(-1), 
                                       &linear_alpha, 
                                       linear_weight_ptr, CUDA_R_32F, linear_weight.size(-1), 
                                       input_tensor_ptr, CUDA_R_32F, input_tensor.size(-1), 
                                       &linear_beta, 
                                       weighted_input_ptr, CUDA_R_32F, linear_weight.size(0), 
                                       CUDA_R_32F, 
                                       static_cast<cublasGemmAlgo_t>(gemm_algo_id));
  // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  // std::cout << "cublasGemmEx status: " << status << std::endl;
  check_cuda_error(status);

  add_bias_input_layernorm_2_kernelLauncher(
    residual_tensor.data_ptr<float>(),
    gamma.data_ptr<float>(),
    beta.data_ptr<float>(),
    linear_bias.data_ptr<float>(),
    weighted_input_ptr,
    output.data_ptr<float>(),
    m,
    n,
    stream);
  
  // std::cout << "++++++++++++ Exit fused_add_layer_norm kernel \n" << std::endl;
  return output;
}


} // namespace torch_ext
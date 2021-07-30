#include "fastertransformer/th_op/topk.h"

#include "fastertransformer/cuda/topk_kernels.cuh"

#include "fastertransformer/cuda/topk_kernels_v2.cuh"

#include "fastertransformer/utils/arguments.h"

#include "fastertransformer/utils/allocator.h"

namespace torch_ext {
using torch::Tensor;

std::vector<Tensor> topK(Tensor input, int64_t k) {
  const std::vector<int64_t> input_sizes = input.sizes().vec();

  float* log_probs = input.data_ptr<float>();

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = k; 
  std::vector<int64_t> temp_sizes = output_sizes;
  const int64_t max_block_per_beam = 8;
  temp_sizes.push_back(max_block_per_beam);
  auto selected_ids = torch::zeros(
    output_sizes,
    torch::dtype(torch::kInt32).device(input.device()).requires_grad(false));
  auto selected_probs = torch::zeros(
    output_sizes,
    torch::dtype(input.dtype()).device(input.device()).requires_grad(false));
  auto temp_log_probs = torch::zeros(
    input_sizes,
    torch::dtype(input.dtype()).device(input.device()).requires_grad(false));
  auto topk_tmp_id_buf = torch::zeros(
    temp_sizes,
    torch::dtype(torch::kInt32).device(input.device()).requires_grad(false));
  auto topk_tmp_val_buf = torch::zeros(
    temp_sizes,
    torch::dtype(input.dtype()).device(input.device()).requires_grad(false));
  
  int* h_ids = selected_ids.data_ptr<int>();
  float* h_values = selected_probs.data_ptr<float>();
  float* temp_log_probs_ptr = temp_log_probs.data_ptr<float>();
  int* topk_tmp_id_buf_ptr = topk_tmp_id_buf.data_ptr<int>();
  float* topk_tmp_val_buf_ptr = topk_tmp_val_buf.data_ptr<float>();

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  fastseq::topK_kernelLauncher<float>(
    log_probs,
    input.numel(),
    input_sizes,
    k,
    h_ids,
    h_values,
    temp_log_probs_ptr,
    topk_tmp_id_buf_ptr,
    topk_tmp_val_buf_ptr,
    stream);
  
  return {selected_probs, selected_ids};
}

}
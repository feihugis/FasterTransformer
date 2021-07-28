#include "fastertransformer/th_op/topk.h"

#include "fastertransformer/cuda/topk_kernels.cuh"

#include "fastertransformer/cuda/topk_kernels_v2.cuh"

#include "fastertransformer/utils/arguments.h"

#include "fastertransformer/utils/allocator.h"

namespace torch_ext {
using torch::Tensor;

Tensor topK(Tensor input, int64_t K) {
  printf("Running topK: %ld * %ld = %ld.... \n", input.size(0), input.size(1), input.numel());

  const int batch_size = input.size(0);
  const int N = input.size(-1); 
  const float* h_log_probs = input.data_ptr<float>();
  const unsigned int num_elems = input.numel();
  const unsigned int mem_size_ids = sizeof(int) * num_elems;
  
  auto ids = torch::zeros({input.size(0), K}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
  int* h_ids = ids.data_ptr<int>();

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));

  // fastertransformer::topK_v2(h_log_probs, h_ids, batch_size, K, N, stream);

  // ids = ids.mode(N);

  // check_cuda_error(cudaMemcpyAsync(d_ids, h_ids, mem_size_ids, cudaMemcpyHostToDevice, stream));

  return ids;
  
  // cudaDeviceSynchronize();
  // check_cuda_error(cudaGetLastError());

  // delete h_ids;
}


std::vector<Tensor> topK_v2(Tensor input, int64_t k) {
  std::unique_ptr<fastertransformer::Allocator<AllocatorType::CUDA>>
    allocator(new fastertransformer::Allocator<AllocatorType::CUDA>(input.device(). index()));
  
  std::vector<int64_t> input_sizes = input.sizes().vec();
  const int vocab_size = input.size(-1);
  const int batch_size = input.numel() / vocab_size;

  int temp_log_probs_buf_size = input.numel();
  int topk_tmp_ids_buf_size = input_sizes[0] * input_sizes[1] * k * 8;
  int topk_tmp_val_buf_size = input_sizes[0] * input_sizes[1] * k * 8;

  // preventing memory misaligned address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  float* log_probs = input.data_ptr<float>();

  std::vector<int64_t> output_sizes = input.sizes().vec();
  output_sizes.back() = k; 
  auto selected_ids = torch::zeros(output_sizes, torch::dtype(torch::kInt32).device(input.device()).requires_grad(false));
  auto selected_probs = torch::zeros(output_sizes, torch::dtype(input.dtype()).device(input.device()).requires_grad(false));
  auto temp_log_probs = torch::zeros(input_sizes, torch::dtype(input.dtype()).device(input.device()).requires_grad(false));
  auto topk_tmp_id_buf = torch::zeros({input_sizes[0], input_sizes[1], k, 8}, torch::dtype(torch::kInt32).device(input.device()).requires_grad(false));
  auto topk_tmp_val_buf = torch::zeros({input_sizes[0], input_sizes[1], k, 8}, torch::dtype(input.dtype()).device(input.device()).requires_grad(false));
  
  int* h_ids = selected_ids.data_ptr<int>();
  float* h_values = selected_probs.data_ptr<float>();
  float* temp_log_probs_ptr = temp_log_probs.data_ptr<float>();
  int* topk_tmp_id_buf_ptr = topk_tmp_id_buf.data_ptr<int>();
  float* topk_tmp_val_buf_ptr = topk_tmp_val_buf.data_ptr<float>();

  void* topk_workspace = (void *)(h_values + selected_probs.numel() * sizeof(float));
  
  size_t workspace_size = sizeof(float) * temp_log_probs_buf_size + 
                          sizeof(int) * topk_tmp_ids_buf_size +
                          2 * sizeof(float) * topk_tmp_val_buf_size;
  // void* topk_workspace = reinterpret_cast<void *>(allocator->malloc(workspace_size, false));
  // cudaDeviceSynchronize();
  
  DecodingBeamsearchArguments args;
  args.batch_size_ = batch_size;
  args.beam_width_ = k;
  args.vocab_size_padded_ = vocab_size;
  args.end_id_ = 50256;
  size_t storage_size_per_beam = 2 * args.beam_width_ + SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2);
  args.temp_storage_size_ = args.batch_size_ * args.beam_width_ * storage_size_per_beam; 
   args.temp_storage_size_ = (size_t)(
      ceil(args.batch_size_ * args.beam_width_ * args.beam_width_ / 4.) * 4 * 2 +
      ceil(args.batch_size_ * args.beam_width_ * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * MAX_K + 2) / 4.) * 4
    );
  args.beam_search_diversity_rate_ = 0.0;
  args.vocab_size_ = vocab_size;

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  fastseq::topK_kernelLauncher<float>(
    topk_workspace, 
    workspace_size,
    log_probs,
    input.numel(),
    h_ids,
    h_values,
    temp_log_probs_ptr,
    topk_tmp_id_buf_ptr,
    topk_tmp_val_buf_ptr,
    nullptr,
    args,
    stream);
  
  // cudaDeviceSynchronize();
  // allocator->free(topk_workspace);
  return {selected_probs, selected_ids};
}

}
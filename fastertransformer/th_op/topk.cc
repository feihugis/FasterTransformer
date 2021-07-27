#include "fastertransformer/th_op/topk.h"

#include "fastertransformer/cuda/topk_kernels.cuh"

#include "fastertransformer/cuda/topk_kernels_v2.cuh"

#include "fastertransformer/utils/arguments.h"

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


Tensor topK_v2(Tensor input, int64_t k) {
  const int vocab_size = input.size(-1);
  const int batch_size = input.size(0) * input.size(1);

  int temp_log_probs_buf_size = input.numel();
  int topk_tmp_ids_buf_size = batch_size * k;
  int topk_tmp_val_buf_size = batch_size * k;

  // preventing memory misaligned address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  float* log_probs = input.data_ptr<float>();

  void* topk_workspace = (void *)(log_probs + input.numel() * 4);
  size_t workspace_size = sizeof(float) * temp_log_probs_buf_size + 
                          sizeof(int) * topk_tmp_ids_buf_size +
                          2 * sizeof(float) * topk_tmp_val_buf_size;
  
  auto ids = torch::zeros({input.size(0), 2, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
  int* h_ids = ids.data_ptr<int>();
  
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
  fastseq::topK_kernelLauncher<float>(topk_workspace, workspace_size, log_probs, h_ids, nullptr, args, stream);
  
  return ids;
}

}
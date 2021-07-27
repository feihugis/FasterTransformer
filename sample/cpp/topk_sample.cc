#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/th_op/topk.h"

#include <cuda_runtime.h>

#include <torch/script.h>
#include <torch/custom_class.h>
#include "torch/csrc/cuda/Stream.h"

#ifdef USE_NVTX
  bool NVTX_ON = true;
#endif

using namespace fastertransformer;

// class TopK : public torch::jit::CustomClassHolder {
//   public:
//     TopK() {}

//     ~TopK() {}

// }

int main(int argc, char* argv[]) {
  printf("Start to run...\n");
  
  int batch_size = 2;
  int beam_width = 1;
  int vocab_size = 5;

  unsigned int mem_size_log_probs = sizeof(float) * batch_size * vocab_size;
  unsigned int mem_size_ids = sizeof(int) * batch_size * vocab_size;

  float h_log_probs[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  int h_ids[10] = {0, 1, 2, 3, 4, 5, 6, 7,8, 9};
  
  float *d_log_probs;
  int *d_ids;
  check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&d_log_probs), mem_size_log_probs));
  check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&d_ids), mem_size_ids));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));

  // copy host memory to device
  check_cuda_error(
      cudaMemcpyAsync(d_log_probs, h_log_probs, mem_size_log_probs, cudaMemcpyHostToDevice, stream));
  check_cuda_error(
      cudaMemcpyAsync(d_ids, h_ids, mem_size_ids, cudaMemcpyHostToDevice, stream));

  fastertransformer::topK(d_log_probs, d_ids, batch_size, beam_width, vocab_size, stream);

  auto input_tensor = torch::from_blob(h_log_probs, {2, 5});
  auto ids = torch_ext::topK_v2(input_tensor, 1);
  cudaDeviceSynchronize();
  check_cuda_error(cudaGetLastError());

  check_cuda_error(
      cudaMemcpyAsync(h_log_probs, d_log_probs, mem_size_log_probs, cudaMemcpyDeviceToHost, stream));
  check_cuda_error(
      cudaMemcpyAsync(h_ids, d_ids, mem_size_ids, cudaMemcpyDeviceToHost, stream));
  
  for(int i = 0; i < 10; ++i) {
    printf("v: %f, id: %d \n", h_log_probs[h_ids[i]], h_ids[i]);
  }

}
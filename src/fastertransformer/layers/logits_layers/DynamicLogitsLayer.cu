#include "src/fastertransformer/layers/logits_layers/DynamicLogitsLayer.h"
#include <cstdlib>

namespace fastertransformer {

__global__ void map_dynamic_vocab_id_2_raw_id(int* dynamic_ids, int* raw_ids_map, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    dynamic_ids[tid] = raw_ids_map[dynamic_ids[tid]];
  }
}


template<typename T>
DynamicLogitsLayer<T>::DynamicLogitsLayer(int vocab_size,
                     int vocab_size_padded,
                     int hidden_size,
                     cublasMMWrapper* cublas_wrapper,
                     IAllocator*      allocator,
                     cudaStream_t     stream) {
    // batch_size_ = batch_size;
    // beam_size_ = beam_size;
    vocab_size_ = vocab_size;
    vocab_size_padded_  = vocab_size_padded;
    hidden_size_ = hidden_size;
    copy_stream_ = stream;
    cublas_wrapper_ = cublas_wrapper;
    allocator_ = allocator;

    dynamic_vocab_size_ = vocab_size_padded;

    // Tensor token_count = Tensor::loadNpy("/dd/fhu/github/FasterTransformer/ads_data/result_v2/tokens_count_v2.npy", MemoryType::MEMORY_CPU);
    // Tensor token_count = Tensor::loadNpy("/dd/fhu/github/FasterTransformer/examples/pytorch/gpt/ads_data/bloom_fliter_raw_data/token_counts_1160136749_int32.npy", MemoryType::MEMORY_CPU);
    std::string token_count_path = std::string(std::getenv("TOKEN_COUNT_PATH"));
    printf("token_count_path: %s\n", token_count_path.c_str());
    Tensor token_count = Tensor::loadNpy(token_count_path, MemoryType::MEMORY_CPU);

    int max_step = token_count.shape[0];
    int max_vocab = token_count.shape[1];
    // max_step = max_gen_len_;
    // max_vocab = vocab_size;
    for (int i = 0; i < max_step; ++i) {
      std::vector<int> indices;
      std::unordered_map<int, int> indices_map;
      for (int j = 0; j < max_vocab; ++j) {
        int token_freq = token_count.getVal<int>(i * max_vocab + j);
        // if (token_freq <= 0) {
        //   // indices.push_back(j);
        //   // indices.push_back(-1);
        //   continue;
        // }
        if (token_freq > 1000 || j == 2) {

          indices.push_back(j);
        }
      }
      dynamic_vocab_indices_.push_back(indices);
    }
    // Tensor token_count = Tensor::loadNpy("/dd/fhu/github/FasterTransformer/ads_data/test.npy", MemoryType::MEMORY_CPU);
    // Tensor token_count = Tensor::loadNpy("/dd/fhu/github/FasterTransformer/tests/data/gpt_context_decoder_inputs/GPU-batch_to_compact_idx.npy", MemoryType::MEMORY_CPU);
    printf("token_count: %s", token_count.toString().c_str());

    for (auto& indices : dynamic_vocab_indices_) {
      auto gpu_indices = (int*)allocator_->malloc(indices.size() * sizeof(int), true/*is_set_zero*/, false/*is_host*/);
      cudaH2Dcpy(gpu_indices, indices.data(), indices.size());
      gpu_dynamic_2_raw_vocab_indices_.push_back(gpu_indices);
    }
  }

template<typename T>
void DynamicLogitsLayer<T>::forward(int step, const T* decoder_outputs, float* logits, int batch_size, int beam_size) {
    float alpha = 1.0f;
    float beta = 0.0f;
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();
    // PUSH_RANGE("dynamic logits copy");
    // cudaStreamWaitEvent(copy_stream_, copy_event_);
    // POP_RANGE;

    PUSH_RANGE("dynamic logits gemm");
    // printf("forward: step %d, vocab size: %d, vocab padded size: %d, dynamic weights ptr: %p \n", step, GetDynamicVocabSize(step), GetDynamicVocabPaddedSize(step), GetDynamicWeightsPtr(step));
    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          GetDynamicVocabPaddedSize(step),  // n
                          batch_size * beam_size,
                          hidden_size_,  // k
                          &alpha,
                          GetDynamicWeightsPtr(step),
                          gemm_data_type,
                          hidden_size_,                                   // k
                          decoder_outputs,  // OPT: no final layer norm
                          gemm_data_type,
                          hidden_size_,  // k
                          &beta,
                          logits,
                          CUDA_R_32F,
                          GetDynamicVocabPaddedSize(step), /* n */
                          CUDA_R_32F,
                          cublasGemmAlgo_t(-1));
    POP_RANGE;
    // cudaDeviceSynchronize();
    // std::string logits_probs = cudaarr2str(logits, GetDynamicVocabPaddedSize(step), dynamic_vocab_indices_[step]);
    // printf("step=%d, logits=%s\n", step, logits_probs.c_str());
    // std::string wte_weight = cudaarr2str<half>(reinterpret_cast<half>(GetDynamicWeightsPtr(step)), 1024);
    // printf("step=%d, wte_weight=%s\n", step, wte_weight.c_str());
}

template<typename T>
void DynamicLogitsLayer<T>::MapDynamicIds2RawIds(int* gpu_dynamic_ids, int step, int num_of_ids) {
  PUSH_RANGE("MapDynamicIds2RawIds");
  int* gpu_dynamic_2_raw_vocab_indices = gpu_dynamic_2_raw_vocab_indices_[step];
  dim3 block(num_of_ids);
  dim3 grid(1);
  map_dynamic_vocab_id_2_raw_id<<<grid, block, 0, copy_stream_>>>(gpu_dynamic_ids, gpu_dynamic_2_raw_vocab_indices, num_of_ids);
  POP_RANGE;
}


template<typename T>
DynamicLogitsLayer<T>::~DynamicLogitsLayer() {
    allocator_->free((void**)(&dynamic_weights_));
    for (auto& ptr: gpu_dynamic_2_raw_vocab_indices_) {
      allocator_->free((void**)(&ptr));
    }
}

template<typename T>
void DynamicLogitsLayer<T>::SetRawWeights(T* weights) {
  weights_ = weights;
  auto dynamic_weights = weights_ + vocab_size_padded_ * hidden_size_;
  // Set all the weights to a very small negative value
  deviceFill<T>(
      dynamic_weights,
      max_gen_len_ * vocab_size_padded_ * hidden_size_,
      -100.0,
      copy_stream_
  );
  FT_CHECK_WITH_INFO(max_gen_len_ <= dynamic_vocab_indices_.size(), "max_gen_len_ should be less than dynamic_vocab_indices_.size()");
  for (int i = 0; i < max_gen_len_; ++i) {
    auto& indices = dynamic_vocab_indices_[i];
    size_t dynamic_vocab_size =  indices.size();
    size_t dynamic_vocab_size_padded = dynamic_vocab_size;

    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
      dynamic_vocab_size_padded = ceil(dynamic_vocab_size_padded / 8.f) * 8;
    }
    dynamic_vocab_sizes_.push_back(dynamic_vocab_size);
    dynamic_vocab_padded_sizes_.push_back(dynamic_vocab_size_padded);
    dynamic_weights_ptrs_.push_back(dynamic_weights);
    printf("step %d, dynamic_vocab_size_padded: %d, dynamic weights ptr: %p \n", i, dynamic_vocab_size_padded, dynamic_weights);
    SetDynamicWeights(dynamic_weights, indices);
    dynamic_weights += vocab_size_padded_ * hidden_size_;
  }

  is_raw_weights_set_ = true;
}

template class DynamicLogitsLayer<float>;
template class DynamicLogitsLayer<half>;
#ifdef ENABLE_BF16
template class DynamicLogitsLayer<__nv_bfloat16>;
#endif

} // namespace fastertransformer
#include "src/fastertransformer/layers/logits_layers/DynamicLogitsLayer.h"

namespace fastertransformer {

__global__ void map_dynamic_vocab_id_2_raw_id(int* dynamic_ids, int* raw_ids_map, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    dynamic_ids[tid] = raw_ids_map[dynamic_ids[tid]];
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
    // printf("forward: step %d, vocab size: %d, dynamic weights ptr: %p \n", step, GetDynamicVocabSize(step), GetDynamicWeightsPtr(step));
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
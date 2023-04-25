#pragma once

#include <vector>


#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/Tensor.h"





namespace fastertransformer {

template<typename T>
class DynamicLogitsLayer {
private:
  // int  batch_size_;
  // int  beam_size_;
  int  vocab_size_;
  int  vocab_size_padded_;
  int  hidden_size_;
  int  dynamic_vocab_size_;      // cpu
  std::vector<std::vector<int>> dynamic_vocab_indices_;  //cpu
  T*   weights_;   // [hidden_size, vocab_size] in row major on GPU
  // T*   logits_;   //  [batch_size, beam_size, vocab_size] in row major on GPU
  T*   dynamic_weights_;  //  [hidden_size, dynamic_vocab_size_] in row major on GPU
  // T*   dynamic_logits_;   //  [batch_size, beam_size, dynamic_vocab_size_] in row major on GPU
  cudaStream_t copy_stream_;
  cudaEvent_t copy_event_;
  cublasMMWrapper* cublas_wrapper_;
  IAllocator*      allocator_;
  bool is_raw_weights_set_ = false;
  size_t max_gen_len_ = 16;
  std::vector<T*> dynamic_weights_ptrs_;
  std::vector<size_t> dynamic_vocab_sizes_;
  std::vector<size_t> dynamic_vocab_padded_sizes_;
  std::vector<int*> gpu_dynamic_2_raw_vocab_indices_;


public:
  DynamicLogitsLayer(int vocab_size,
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

    Tensor token_count = Tensor::loadNpy("/dd/fhu/github/FasterTransformer/ads_data/result_v2/tokens_count.npy", MemoryType::MEMORY_CPU);
    int max_step = token_count.shape[0];
    int max_vocab = token_count.shape[1];
    for (int i = 0; i < max_step; ++i) {
      std::vector<int> indices;
      std::unordered_map<int, int> indices_map;
      for (int j = 0; j < max_vocab; ++j) {
        int token_freq = token_count.getVal<int>(i * max_vocab + j);
        if (token_freq == 0) {
          // indices.push_back(j);
          // indices.push_back(-1);
          continue;
        }
        if (token_freq > 0) {

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

  void MapDynamicIds2RawIds(int* gpu_dynamic_ids, int step, int number_of_ids);

  void SetRawWeights(T* weights);

  bool IsRawWeightsSet() {
    return is_raw_weights_set_;
  }

  size_t GetMaxGenLen() {
    return max_gen_len_;
  }

  int GetRawVocabId(int step, int dynamic_vocab_id) {
    return dynamic_vocab_indices_[step][dynamic_vocab_id];
  }

  T* GetDynamicWeightsPtr(int step) {
    return dynamic_weights_ptrs_[step];
  }

  size_t GetDynamicVocabSize(int step) {
    return dynamic_vocab_sizes_[step];
  }

  size_t GetDynamicVocabPaddedSize(int step) {
    return dynamic_vocab_padded_sizes_[step];
  }

  void SetDynamicWeights(T* dynamic_weights, const std::vector<int>&  dynamic_vocab_indices) {
    PUSH_RANGE("DynamicLogitsLayer SetDynamicWeights");
    int i = 0;
    while (i < dynamic_vocab_indices.size()) {
      int start = dynamic_vocab_indices[i];
      if (start == -1) {
        ++i;
        continue;
      }
      int end = dynamic_vocab_indices[i];
      int j = i;
      while (j < dynamic_vocab_indices.size()) {
        end = dynamic_vocab_indices[j];
        if (end == -1) {
          ++j;
          break;
        }
        if (end - start != j - i) {
          break;
        }
        ++j;
      }
      // printf("%p copy from %d to %d, size: %d \n", dynamic_weights, start, end, (j - i) * hidden_size_);
      cudaMemcpyAsync(
        dynamic_weights + i * hidden_size_,
        weights_ + start * hidden_size_,
        (j - i) * hidden_size_ * sizeof(T),
        cudaMemcpyDeviceToDevice, copy_stream_);
      // cudaMemcpy(
      //   dynamic_weights + i * hidden_size_,
      //   weights_ + start * hidden_size_,
      //   (j - i) * hidden_size_ * sizeof(T),
      //   cudaMemcpyDeviceToDevice);
      i = j;
    }

    // deviceFill<T>(
    //     dynamic_weights_ + 20000 * hidden_size_,
    //     20000 * hidden_size_,
    //     -10.0,
    //     copy_stream_
    //     );

    // cudaEventRecord(copy_event_, copy_stream_);
    POP_RANGE;
  }

  ~DynamicLogitsLayer();


  void forward(int step, const T* decoder_outputs, float* logits, int batch_size, int beam_size);
};
} // namespace fastertransformer

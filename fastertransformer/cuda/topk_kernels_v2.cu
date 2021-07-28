/*
* Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "fastertransformer/cuda/topk_kernels_v2.cuh"
#include "fastertransformer/cuda/topk_kernels.cuh"
#include "cub/cub.cuh"

namespace fastseq {

  using namespace fastertransformer;

  template void topK_kernelLauncher<float>(void* workspace,
    size_t& workspace_size,
    float* log_probs,
    int* ids,
    float* values,
    const bool* finished,
    DecodingBeamsearchArguments args,
    cudaStream_t stream);

  template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
  __global__ void topk_stage_1_opt2_general(
      const T* __restrict log_probs,
      T* tmp_log_probs,
      int* topk_tmp_id_buf,
      T* topk_tmp_val_buf,
      const int k,
      const int vocab_size
  )
  {
      const bool IS_FP16 = std::is_same<T, half>::value;
      const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
      typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int row_id = bid / BLOCKS_PER_BEAM; // row id for log_probs
      const int block_lane = bid % BLOCKS_PER_BEAM; // block id for a beam 
      const int tmp_log_buf_index = row_id * vocab_size; 
      const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
      TopK_2<T> partial;
  
      for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
      {
          int index = elem_id + tmp_log_buf_index;
          tmp_log_probs[index] = log_probs[index]; 
      }
  
  
      for(int ite = 0; ite < k; ite++)
      {
          partial.init();
          #pragma unroll
          for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
          {
              int index = elem_id + tmp_log_buf_index;
              partial.insert(tmp_log_probs[index], index);
          }
  
          TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, fastertransformer::reduce_topk_op_2<T>);
  
          if (tid == 0)
          {
              const int index = tmp_topk_buf_index + ite;
              topk_tmp_id_buf[index] = total.p;
              topk_tmp_val_buf[index] = total.u;
              tmp_log_probs[total.p] = -MAX_T_VAL;
          }
          __syncthreads();
      }
  }
  
  template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
  __global__ void topk_stage_2_opt2_general(
      const int* __restrict topk_tmp_id_buf,
      T* topk_tmp_val_buf,
      int* ids,
      const int k)
  {
      const int size = k * k * BLOCKS_PER_BEAM; 
      const int tid = threadIdx.x;
      const int batch_id = blockIdx.x;
      const bool IS_FP16 = std::is_same<T, half>::value;
      const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;
  
      typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      extern __shared__ char array[];
      T *s_val = topk_tmp_val_buf + batch_id * size;
      int *s_id = (int*)(array);
      
      TopK_2<T> partial;
  
      for(int ite = 0; ite < k; ite++)
      {
          partial.init();
          #pragma unroll
          for(int i = tid; i < size; i+= BLOCK_SIZE)
          {
              partial.insert(s_val[i], i);
          }
      
          TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);
      
          if(tid == 0) 
          {
              s_id[ite] = total.p;
              s_val[total.p] = -MAX_T_VAL;
          }
          __syncthreads();
      }
      if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
  }

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const bool* finished,
    const int k,
    const int vocab_size,
    const int end_id
)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size; 
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK_2<T> partial;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    if(finished != nullptr && finished[row_id] == true)
    {
        if(tid < k)
        {
            const int index = tmp_topk_buf_index + tid;
            if(block_lane == 0 && tid == 0)
            {
                topk_tmp_id_buf[index] = tmp_log_buf_index + end_id;
                topk_tmp_val_buf[index] = log_probs[tmp_log_buf_index + end_id]; 
            }
            else
            {
                topk_tmp_id_buf[index] = -1;
                topk_tmp_val_buf[index] = -MAX_T_VAL; 
                
            }
        }
        return;
    }

    for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index]; 
        // printf("+++ tid: %d, bid: %d, row_id: %d, block_lane: %d, index: %d, value: %f\n", tid, bid, row_id, block_lane, index, log_probs[index]);
    }

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
            // printf("+++ tid: %d, bid: %d, row_id: %d, block_lane: %d, index: %d, value: %f\n", tid, bid, row_id, block_lane, index, tmp_log_probs[index]);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -MAX_T_VAL;
            // printf("+++ Top-%d, tid: %d, bid: %d, row_id: %d, block_lane: %d, index: %d, value: %f\n", ite, tid, bid, row_id, block_lane, total.p, total.u);
        }

        // printf("+++ tid: %d, bid: %d, row_id: %d, block_lane: %d, index: %d, value: %f\n", tid, bid, row_id, block_lane, total.p, total.u);
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(
    const T* __restrict log_probs,
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    T* values,
    const int k,
    const int vocab_size)
{
    const int size = k * BLOCKS_PER_BEAM_; 
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16)? HALF_FLT_MAX : FLT_MAX;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);
    
    fastertransformer::TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE_)
        {
            partial.insert(s_val[i], i);
        }
    
        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, fastertransformer::reduce_topk_op_2<T>);
    
        if(tid == 0) 
        {
            s_id[ite] = total.p;
            // printf("tid: %d, batch_id: %d, index: %d, val: %f \n", tid, batch_id, total.p, s_val[total.p]);
            s_val[total.p] = -MAX_T_VAL;
        }

        // printf("+++ tid: %d, bid: %d, row_id: %d, block_lane: %d, index: %d, value: %d\n", tid, batch_id, row_id, block_lane, total.p, total.u);
        __syncthreads();
    }
    if(tid < k) {
      ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
      values[batch_id * k + tid] = log_probs[ids[batch_id * k + tid]];
      ids[batch_id * k + tid] = ids[batch_id * k + tid] - (ids[batch_id * k + tid] / vocab_size) * vocab_size;
    }
}

#define CASE_K(K, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K: \
    topk_stage_1_opt3<float, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<batch_size * K * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
        log_probs, \
        temp_log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        finished, \
        beam_width, vocab_size, end_id); \
    topk_stage_2_opt3<float, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, K * sizeof(int), stream>>>( \
        log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        ids, \
        values, \
        beam_width, \
        vocab_size); \
  break; \

template <typename T>
void topK_kernelLauncher(void* workspace,
                         size_t& workspace_size,
                         T* log_probs,
                         int* ids,
                         T* values,
                         const bool* finished,
                         fastertransformer::DecodingBeamsearchArguments args,
                         cudaStream_t stream)
{
    const int batch_size = args.batch_size_;
    const int beam_width = args.beam_width_;
    const int vocab_size = args.vocab_size_padded_;
    const T diversity_rate = args.beam_search_diversity_rate_;
    const int end_id = args.end_id_;

    const int max_block_per_beam = 8;
    int temp_log_probs_buf_size = batch_size * beam_width * vocab_size; // type float
    int topk_tmp_ids_buf_size = batch_size * beam_width * beam_width * max_block_per_beam;      // type int
    int topk_tmp_val_buf_size = batch_size * beam_width * beam_width * max_block_per_beam;      // type float

    // prevent memory misalinged address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;
    
    if(workspace == nullptr)
    {
        workspace_size = sizeof(float) * temp_log_probs_buf_size + 
                         sizeof(int) * topk_tmp_ids_buf_size + 
                         sizeof(float) * topk_tmp_val_buf_size;
        return;
    }
    else
    {
        T* temp_log_probs = (T*)workspace;
        int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
        T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);
        if(diversity_rate == 0.0f)
        {
            switch(beam_width)
            {
                CASE_K(1,128,128,8);
                CASE_K(4,128,128,8);
                CASE_K(10,128,128,8);
                CASE_K(16,128,128,5);
                CASE_K(32,256,128,1);
                CASE_K(64,256,256,1);
                default:
                    topk_stage_1_opt2_general<T, 128, 1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
                        log_probs,
                        temp_log_probs,
                        topk_tmp_id_buf,
                        topk_tmp_val_buf,
                        beam_width, vocab_size);
                    topk_stage_2_opt2_general<T, 128, 1><<<batch_size, 128, 
                            beam_width*beam_width*1*sizeof(float) + beam_width * sizeof(int), stream>>>(
                        topk_tmp_id_buf,
                        topk_tmp_val_buf,
                        ids,
                        beam_width);
                    break;
            }
        }
        return;
    }
}

#undef CASE_K

} //namespace fastseq

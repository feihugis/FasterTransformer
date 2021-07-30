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
#pragma once
#include <assert.h>
#include <array>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "fastertransformer/utils/arguments.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include <float.h>
#include <type_traits>

#include <vector>

namespace fastseq {

template <typename T>
void topK_kernelLauncher(T* log_probs,
                         const size_t num_elements,
                         const std::vector<int64_t>& input_sizes,
                         const int64_t k,
                         int* ids,
                         T* values,
                         T* temp_log_probs,
                         int* topk_tmp_id_buf,
                         T* topk_tmp_val_buf,
                         cudaStream_t stream);

} //namespace fastseq
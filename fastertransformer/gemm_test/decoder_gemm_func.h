#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include <map>

namespace fastertransformer{

template<typename T>
void generate_gemm_config(int batchcount,
                          int m,
                          int n,
                          int k,
                          bool is_append=true);

}

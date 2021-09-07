#ifndef __ONE_HOT_KERNEL__H_
#define __ONE_HOT_KERNEL__H_

#include "NvInferPlugin.h"
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_fp16.h>

using half = __half;

void computeOneHot(const int32_t* input, const int32_t* depth, const half* values, half* output, cudaStream_t stream);

void computeOneHot(const int32_t* input, const int32_t* depth, const float* values, float* output, cudaStream_t stream);

#endif

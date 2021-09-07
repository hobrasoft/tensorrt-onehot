#include "onehotkernel.h"


__global__ void gpuOneHot(const int32_t* input, const int32_t* depth, const half* values, half* output, cudaStream_t stream) {
    for (int i=0; i<*depth; i++) {
        output[i] = values[0];
        }
    output[*input] = values[1];
}


__global__ void gpuOneHot(const int32_t* input, const int32_t* depth, const float *values, float* output, cudaStream_t stream) {
    for (int i=0; i<*depth; i++) {
        output[i] = values[0];
        }
    output[*input] = values[1];
}


void computeOneHot(const int32_t* input, const int32_t* depth, const half* values, half* output, cudaStream_t stream) {
    gpuOneHot<<<1, 1, 0, stream>>>(input, depth, values, output, stream);
}


void computeOneHot(const int32_t* input, const int32_t* depth, const float *values, float* output, cudaStream_t stream) {
    gpuOneHot<<<1, 1, 0, stream>>>(input, depth, values, output, stream);
}




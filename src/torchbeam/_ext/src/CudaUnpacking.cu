#include <cuda_runtime.h>
#include "../include/Unpacking.h"

__global__ void cudaUnpack10to16kernel(uint16_t* dest, const uint8_t* src, int intCount) {
    
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if ((threadId+1) * 4 >= intCount) {
        return;
    }

    src += threadId * 5;
    dest += threadId * 4;

    uint32_t val = 0;
    val |= *(src++) << 0;
    val |= *(src++) << 8;
    val |= *(src++) << 16;
    val |= *(src++) << 24;

    *dest++ = (val >>  0) & 0x3FF;
    *dest++ = (val >> 10) & 0x3FF;
    *dest++ = (val >> 20) & 0x3FF;
    *dest = (val >> 30) & 0x003 | *src << 2;
}

void cudaUnpack10to16(void* dest, void* src, int intCount) {
    int blockSize = 256;
    int numBlocks = ((intCount / 4) + blockSize - 1) / blockSize;
    cudaUnpack10to16kernel<<<numBlocks, blockSize>>>((uint16_t*)dest, (uint8_t*)src, intCount);
}

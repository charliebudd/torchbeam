#include <cuda_runtime.h>
#include "../include/CudaConvert.h"

__global__ void gray10le_to_p010le_kernel(uint16_t* dst, int dstPitch, const uint16_t* src, int srcPitch, int srcWidth, int srcHeight) {
    
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= srcWidth * srcHeight) {
        return;
    }

    int y = threadId / srcWidth;
    int x = threadId - y * srcWidth;
    
    int srcId = y * srcPitch + x;
    int dstId = y * dstPitch + x;

    dst[dstId] = src[srcId] << 6;
}

void gray10le_to_p010le(uint16_t* dst, int dstPitch, const uint16_t* src, int srcPitch, int srcWidth, int srcHeight) {
    int blockSize = 256;
    int numBlocks = ((srcWidth * srcHeight) + blockSize - 1) / blockSize;
    gray10le_to_p010le_kernel<<<numBlocks, blockSize>>>(dst, dstPitch, src, srcPitch, srcWidth, srcHeight);
}

#include "../include/TorchTexture.h"

void checkCudaError(cudaError_t cudaStatus, const char* errorMessage) {
    if (cudaStatus != cudaSuccess) {
        std::cerr << errorMessage << " CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

TorchTexture::TorchTexture(int glTextureID) {
    if (glIsTexture(glTextureID) == GL_FALSE) {
        std::cerr << "Invalid OpenGL texture ID" << std::endl;
    }

    cudaError_t cudaStatus = cudaGraphicsGLRegisterImage(&m_cudaGraphicsResource, glTextureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    checkCudaError(cudaStatus, "Failed to register OpenGL image resource");

    cudaStatus = cudaGraphicsMapResources(1, &m_cudaGraphicsResource);
    checkCudaError(cudaStatus, "Failed to map CUDA resources");

    cudaStatus = cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_cudaGraphicsResource, 0, 0);
    checkCudaError(cudaStatus, "Failed to get mapped CUDA array");
}

TorchTexture::~TorchTexture() {
    cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource);
    cudaGraphicsUnregisterResource(m_cudaGraphicsResource);
}

void TorchTexture::blitTexture(torch::Tensor tensor) {
    int byteWidth =  tensor.size(2) * tensor.size(1) * sizeof(uint8_t);
    glFinish();
    cudaMemcpy2DToArray(m_cudaArray, 0, 0, tensor.contiguous().data_ptr(), byteWidth, byteWidth, tensor.size(0), cudaMemcpyDeviceToDevice);
}

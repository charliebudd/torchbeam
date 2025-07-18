#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <iostream>

class TorchTexture {
public:
    TorchTexture(int glTextureID);
    ~TorchTexture();
    
    void blitTexture(torch::Tensor tensor);

private:
    cudaGraphicsResource_t m_cudaGraphicsResource;
    cudaArray_t m_cudaArray;
};

#include <cuda_runtime.h>

#include "../include/TorchCamera.h"
#include "../include/Unpacking.h"

TorchCamera::TorchCamera(PixelFormat pixelFormat, bool useCuda, ICamera* camera) :
m_pixelFormat(pixelFormat), m_useCuda(useCuda), m_camera(camera) {

    m_width = m_camera->getIntAttribute("Width");
    m_height = m_camera->getIntAttribute("Height");
    m_payloadSize = m_camera->getIntAttribute("PayloadSize");

    // Rounding up to nearest 64 bits for wiggle room when unpacking
    m_payloadSize = (64 * ((63 + m_payloadSize * 8) / 64)) / 8;

    auto dtype = pixelFormatIs8Bit(m_pixelFormat) ? torch::kUInt8 : torch::kInt16;
    auto device = m_useCuda ? torch::kCUDA : torch::kCPU;
    m_imageTensor = torch::empty({m_height, m_width}, torch::dtype(dtype).device(device));

    for (int i=0; i<2; i++) {
        if (m_useCuda) {
            cudaHostAlloc(&m_buffers[i], m_payloadSize, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        }
        else {
            m_buffers[i] = malloc(m_payloadSize);
        }
    }

    m_camera->setBuffers(m_buffers, 2);
    m_camera->startAcquisition();
}

TorchCamera::~TorchCamera() {
    m_camera->stopAcquisition();
    delete m_camera;

    for (int i=0; i<2; i++) {
        if (m_useCuda) {
            cudaFreeHost(m_buffers[i]);
        }
        else {
            free(m_buffers[i]);
        }
    }
}


bool TorchCamera::imageReady() {
    return m_camera->bufferReady();
}

std::pair<torch::Tensor, uint64_t> TorchCamera::getImage() {

    std::pair<void*, uint64_t> frame = m_camera->getBuffer();

    if (m_useCuda) {
        void* deviceAccessibleBuffer;
        cudaHostGetDevicePointer(&deviceAccessibleBuffer, frame.first, 0);

        if (pixelFormatIs8Bit(m_pixelFormat)) {
            cudaMemcpy(m_imageTensor.data_ptr(), deviceAccessibleBuffer, m_payloadSize, cudaMemcpyDeviceToDevice);
        }
        else {
            int intCount = m_width * m_height;
            cudaUnpack10to16(m_imageTensor.data_ptr(), deviceAccessibleBuffer, intCount);
        }
    }
    else {
        if (pixelFormatIs8Bit(m_pixelFormat)) {
            memcpy(m_imageTensor.data_ptr(), frame.first, m_payloadSize);
        }
        else {
            int intCount = m_width * m_height;
            cpuUnpack10to16((uint16_t*)m_imageTensor.data_ptr(), (uint8_t*)frame.first, intCount);
        }
    }

    return std::pair(m_imageTensor, frame.second);
}

float TorchCamera::getFloatAttribute(std::string name) {
    return m_camera->getFloatAttribute(name);
}
void TorchCamera::setFloatAttribute(std::string name, float value) {
    m_camera->setFloatAttribute(name, value);
}

int TorchCamera::getIntAttribute(std::string name) {
    return m_camera->getIntAttribute(name);
}

void TorchCamera::setIntAttribute(std::string name, int value) {
    m_camera->setIntAttribute(name, value);
}

#pragma once

#include <torch/extension.h>
#include <functional>

#include "ICamera.h"
#include "PixelFormat.h"


class TorchCamera {

public:

    TorchCamera(PixelFormat pixelFormat, bool useCuda, ICamera* camera);
    ~TorchCamera();

    std::pair<torch::Tensor, uint64_t> getImage();

    bool imageReady();

    float getFloatAttribute(std::string name);
    void setFloatAttribute(std::string name, float value);
    int getIntAttribute(std::string name);
    void setIntAttribute(std::string name, int value);

private:

    PixelFormat m_pixelFormat;
    bool m_useCuda;
    ICamera* m_camera;
    void* m_buffers[3];
    torch::Tensor m_imageTensor;
    int m_payloadSize, m_width, m_height;

};

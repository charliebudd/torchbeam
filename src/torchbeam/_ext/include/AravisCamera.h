#pragma once

#include <arv.h>

#include "ICamera.h"
#include "PixelFormat.h"

class AravisCamera : public ICamera {

public:

    AravisCamera(PixelFormat pixelFormat);
    ~AravisCamera();

    void startAcquisition() override;
    void stopAcquisition() override;

    void setBuffers(void** buffers, int count) override;
    
    bool bufferReady() override;
    std::pair<void*, uint64_t> getBuffer() override;

    float getFloatAttribute(std::string name) override;
    void setFloatAttribute(std::string name, float value) override;

    int getIntAttribute(std::string name) override;
    void setIntAttribute(std::string name, int value) override;

    void onImageReady(ArvStreamCallbackType type, ArvBuffer* buffer);
    
private:

    PixelFormat m_pixelFormat;
    ArvCamera* m_camera;
    ArvStream* m_stream;
    ArvBuffer* m_readBuffer;
    bool m_bufferReady = false;
    bool m_done = false;

};

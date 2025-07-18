#pragma once
#include <string>
#include <functional>

using ImageReadyCallback = std::function<void()>;

class ICamera {

public:

    virtual ~ICamera() {}

    virtual void startAcquisition() = 0;
    virtual void stopAcquisition() = 0;

    virtual void setBuffers(void** buffers, int count) = 0;
    
    virtual bool bufferReady() = 0;
    virtual std::pair<void*, uint64_t> getBuffer() = 0;

    virtual float getFloatAttribute(std::string name) = 0;
    virtual void setFloatAttribute(std::string name, float value) = 0;

    virtual int getIntAttribute(std::string name) = 0;
    virtual void setIntAttribute(std::string name, int value) = 0;
    
};


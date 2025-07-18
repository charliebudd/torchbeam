#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>

#include "ICamera.h"
#include "PixelFormat.h"

class FakeCamera : public ICamera {

public:

    FakeCamera(PixelFormat pixelFormat, int width, int height);
    ~FakeCamera();

    void startAcquisition() override;
    void stopAcquisition() override;

    void setBuffers(void** buffers, int count) override;
    
    bool bufferReady() override;
    std::pair<void*, uint64_t> getBuffer() override;

    void writeToBuffers();

    float getFloatAttribute(std::string name) override;
    void setFloatAttribute(std::string name, float value) override;

    int getIntAttribute(std::string name) override;
    void setIntAttribute(std::string name, int value) override;

private:

    PixelFormat m_pixelFormat;
    int m_height, m_width;

    bool m_running;
    std::thread m_writerThread;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::queue<void*> m_buffers;
    void* m_writeBuffer;
    void* m_readyBuffer;
    void* m_readBuffer;

    float m_exposure = 8000;
    float m_gain = 1.0;
};

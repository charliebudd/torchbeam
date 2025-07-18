#include <chrono>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <algorithm>

#include "../include/FakeCamera.h"

static int64_t getCurrentTimeNanoSeconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

void generateData8Bit(uint8_t* buffer, int width, int height, float scale, int64_t t) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dist = int(sqrt(x * x + y * y));
            int value = int(scale * (dist + t / 5000000)) % 500;
            value = 255 * ((1+((y+x)%4)) / 4.0f) * (value / 500.0f);
            value = std::clamp(int(scale * value), 0, 255);
            buffer[x + y * width] = value;
        }
    }
}

void generateData10Bit(uint8_t* buffer, int width, int height, float scale, int64_t t) {
    int intCount = 0;
    uint64_t values = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dist = int(sqrt(x * x + y * y));
            uint64_t value = (dist + t / 5000000) % 500;
            value = 1023 * ((1+((y+x)%4)) / 4.0f) * (value / 500.0f);
            value = std::clamp(int(scale * value), 0, 1023);
            values |= (value & 0x3FF) << (intCount * 10);
            intCount++;
            if (intCount == 4) {
                *((uint64_t*)buffer) = 0xFFFFFFFFFF & values;
                values = 0;
                intCount = 0;
                buffer += 5;
            }
        }
    }
}

FakeCamera::FakeCamera(PixelFormat pixelFormat, int height, int width) :
m_pixelFormat(pixelFormat), m_height(height), m_width(width), m_running(false), m_writeBuffer(nullptr), m_readyBuffer(nullptr), m_readBuffer(nullptr) {
    
}

FakeCamera::~FakeCamera() {

}

void FakeCamera::startAcquisition() {
    m_running = true;
    m_writerThread = std::thread(&FakeCamera::writeToBuffers, this);
}

void FakeCamera::stopAcquisition() {
    m_running = false;
    m_cv.notify_all();
    if (m_writerThread.joinable()) {
        m_writerThread.join();
    }
}

void FakeCamera::setBuffers(void** buffers, int count) {
    for (int i=0; i<count; i++) {
        m_buffers.push(buffers[i]);
    }
    m_writeBuffer = m_buffers.front();
    m_buffers.pop();
    m_readBuffer = nullptr;
    m_readyBuffer = nullptr;
}

bool FakeCamera::bufferReady() {

    std::unique_lock lock(m_mtx);

    if (m_readBuffer != nullptr) {
        m_buffers.push(m_readBuffer);
        m_readBuffer = nullptr;
        m_cv.notify_all();
    }
    
    bool ready = m_readyBuffer != nullptr;

    return ready;
}


std::pair<void*, uint64_t> FakeCamera::getBuffer() {

    std::unique_lock lock(m_mtx);

    if (m_readBuffer != nullptr) {
        m_buffers.push(m_readBuffer);
        m_readBuffer = nullptr;
        m_cv.notify_all();
    }

    m_cv.wait(lock, [this]{ return this->m_readyBuffer != nullptr; });
    m_readBuffer = m_readyBuffer;
    m_readyBuffer = nullptr;

    uint64_t timestamp = getCurrentTimeNanoSeconds();
    return std::pair(m_readBuffer, timestamp);
}

void FakeCamera::writeToBuffers() {
    while (m_running) {

        int64_t t = getCurrentTimeNanoSeconds();

        std::this_thread::sleep_for(std::chrono::microseconds(int(m_exposure)));

        float scale = pow(10, m_gain/20) * m_exposure / 1e5;

        if (pixelFormatIs8Bit(m_pixelFormat)) {
            generateData8Bit((uint8_t*)m_writeBuffer, m_width, m_height, scale, t);
        }
        else {
            generateData10Bit((uint8_t*)m_writeBuffer, m_width, m_height, scale, t);
        }
        
        {
            std::unique_lock lock(m_mtx);
            if (m_readyBuffer != nullptr) {
                m_buffers.push(m_readyBuffer);
            }
            m_readyBuffer = m_writeBuffer;
            m_writeBuffer = nullptr;
        }

        m_cv.notify_all();

        {
            std::unique_lock lock(m_mtx);
            m_cv.wait(lock, [this]{ return !this->m_running || !this->m_buffers.empty(); });

            if (m_running) {
                m_writeBuffer = m_buffers.front();
                m_buffers.pop();
            }
        }
    }
}

float FakeCamera::getFloatAttribute(std::string name) {
    if (name == "ExposureTime") {
        return m_exposure;
    }
    else if (name == "Gain") {
        return m_gain;
    }
    else {
        return 42.0;
    }
}

void FakeCamera::setFloatAttribute(std::string name, float value) {
    if (name == "ExposureTime") {
        m_exposure = value;
    }
    else if (name == "Gain") {
        m_gain = value;
    }
}

int FakeCamera::getIntAttribute(std::string name) {
    if (name == "Width") {
        return m_width;
    }
    else if (name == "Height") {
        return m_height;
    }
    else if (name == "PayloadSize") {
        return m_width * m_height * pixelFormatBitCount(m_pixelFormat) / 8;
    }
    else {
        return 42;
    }
}

void FakeCamera::setIntAttribute(std::string name, int value) {
    
}

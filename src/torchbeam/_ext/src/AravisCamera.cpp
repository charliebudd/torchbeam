#include <stdexcept>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream>
#include <libusb.h>

#include "../include/AravisCamera.h"

static GError* error = NULL;
void checkError(const char* message) {
    if (error != NULL) {
        std::cerr << message << std::endl;
        throw std::runtime_error(error->message);
    }
}

static void arvOnImageRead(void* userdata, ArvStreamCallbackType type, ArvBuffer* buffer) {
    AravisCamera* cam = (AravisCamera*)userdata;
    cam->onImageReady(type, buffer);
}

AravisCamera::AravisCamera(PixelFormat pixelFormat) : m_pixelFormat(pixelFormat), m_readBuffer(nullptr) {

    arv_update_device_list();
    int deviceCount = arv_get_n_devices();
    if (deviceCount != 1) {
        std::ostringstream oss;
        oss << "Found " << deviceCount << " connected cameras.";
        throw std::runtime_error(oss.str());
    }

    m_camera = arv_camera_new(NULL, &error);

    if (error != NULL) {
        g_clear_object(&m_camera);
        std::cerr << "Camera creation" << std::endl;
        throw std::runtime_error(error->message);
    }

	arv_camera_set_acquisition_mode(m_camera, ARV_ACQUISITION_MODE_CONTINUOUS, &error);
    checkError("Setting acquisition mode");
	arv_camera_set_pixel_format_from_string(m_camera, pixelFormatName(m_pixelFormat), &error);
    checkError("Setting pixel format");

    ArvUvDevice* uvDevice = ARV_UV_DEVICE(G_OBJECT(arv_camera_get_device(m_camera)));
    arv_uv_device_set_usb_mode(uvDevice, ARV_UV_USB_MODE_ASYNC);
}

AravisCamera::~AravisCamera() {
    m_done = true;
    g_clear_object(&m_stream);
    g_clear_object(&m_camera);
}

void AravisCamera::startAcquisition() {
    arv_camera_start_acquisition(m_camera, &error);
    checkError("Starting");
}

void AravisCamera::stopAcquisition() {
    arv_camera_stop_acquisition(m_camera, &error);
    checkError("Stoping");
}

void AravisCamera::setBuffers(void** buffers, int count) {
	size_t payloadSize = arv_camera_get_payload(m_camera, &error);
    checkError("Getting payload size");
    m_stream = arv_camera_create_stream(m_camera, &arvOnImageRead, (void*)this, &error);
    checkError("Creating stream");
    arv_stream_push_buffer(m_stream, arv_buffer_new(payloadSize, buffers[0]));
    m_readBuffer = arv_buffer_new(payloadSize, buffers[1]);
}

bool AravisCamera::bufferReady() {
    return m_bufferReady;
}

std::pair<void*, uint64_t> AravisCamera::getBuffer() {

    ArvBuffer* oldBuffer = m_readBuffer;
    m_readBuffer = arv_stream_pop_buffer(m_stream);
    arv_stream_push_buffer(m_stream, oldBuffer);

    void* data = const_cast<void*>(arv_buffer_get_data(m_readBuffer, NULL));
    uint64_t timestamp = arv_buffer_get_system_timestamp(m_readBuffer);

    m_bufferReady = false;

    return std::pair(data, timestamp);
}

void AravisCamera::onImageReady(ArvStreamCallbackType type, ArvBuffer* buffer) {
    if (!m_done && type == ARV_STREAM_CALLBACK_TYPE_BUFFER_DONE) {
        if (m_bufferReady) {
            std::cout << "Frame Dropped." << std::endl;
        }
        m_bufferReady = true;
    }
}

float AravisCamera::getFloatAttribute(std::string name) {
    float value = arv_camera_get_float(m_camera, name.c_str(), &error);
    checkError(name.c_str());
    return value;
}

void AravisCamera::setFloatAttribute(std::string name, float value) {
    arv_camera_set_float(m_camera, name.c_str(), value, &error);
    checkError(name.c_str());
}

int AravisCamera::getIntAttribute(std::string name) {
    int value = arv_camera_get_integer(m_camera, name.c_str(), &error);
    checkError(name.c_str());
    return value;
}

void AravisCamera::setIntAttribute(std::string name, int value) {
    arv_camera_set_integer(m_camera, name.c_str(), value, &error);
    checkError(name.c_str());
}

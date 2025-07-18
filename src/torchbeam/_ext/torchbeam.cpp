#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "include/PixelFormat.h"
#include "include/AravisCamera.h"
#include "include/FakeCamera.h"
#include "include/TorchCamera.h"
#include "include/TorchTexture.h"
#include "include/TorchVideoEncoder.h"
#include "include/TorchVideoDecoder.h"


// Casts python int to PixelFormat
template <> struct py::detail::type_caster<PixelFormat> {
public:
    PYBIND11_TYPE_CASTER(PixelFormat, _("PixelFormat"));
    bool load(handle src, bool convert) {
        if (!convert && !PyLong_Check(src.ptr())) {
            return false;
        }
        value = static_cast<PixelFormat>(PyLong_AsLong(src.ptr()));
        return true;
    }
};

// Just to allow spoofing the constructor..
class TorchCameraBinding : public TorchCamera {

public:

    TorchCameraBinding(PixelFormat pixelFormat, bool useCuda, bool fake) :
    TorchCamera(
        pixelFormat, useCuda,
        fake ? (ICamera*)(new FakeCamera(pixelFormat, 1080, 1920)) : (ICamera*)(new AravisCamera(pixelFormat))
    ) {}

};

PYBIND11_MODULE(torchbeam_ext, m) {
    
    m.attr("PIXEL_FORMAT_MONO8") = py::int_(static_cast<int>(PixelFormat::Mono8));
    m.attr("PIXEL_FORMAT_MONO10p") = py::int_(static_cast<int>(PixelFormat::Mono10p));

    pybind11::class_<TorchCameraBinding>(m, "Camera")
        .def(pybind11::init<PixelFormat, bool, bool>())
        .def("getImage", &TorchCameraBinding::getImage)
        .def("imageReady", &TorchCameraBinding::imageReady)
        .def("getFloatAttribute", &TorchCameraBinding::getFloatAttribute)
        .def("setFloatAttribute", &TorchCameraBinding::setFloatAttribute)
        .def("getIntAttribute", &TorchCameraBinding::getIntAttribute)
        .def("setIntAttribute", &TorchCameraBinding::setIntAttribute);

    pybind11::class_<TorchTexture>(m, "TorchTexture")
        .def(pybind11::init<int>())
        .def("blitTexture", &TorchTexture::blitTexture);

    pybind11::class_<TorchVideoEncoder>(m, "TorchVideoEncoder")
        .def(pybind11::init<std::string, int, int, int, std::string, std::string, std::string>())
        .def("writeFrame", &TorchVideoEncoder::writeFrame)
        .def("close", &TorchVideoEncoder::close);

    pybind11::class_<TorchVideoDecoder>(m, "TorchVideoDecoder")
        .def(pybind11::init<std::string, std::string>())
        .def("getMetadata", &TorchVideoDecoder::getMetadata)
        .def("readFrame", &TorchVideoDecoder::readFrame)
        .def("close", &TorchVideoDecoder::close);
}

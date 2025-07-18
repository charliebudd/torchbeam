import torch
from enum import Enum
from typing import Tuple

import torchbeam_ext as ext

__all__ = ["PixelFormat", "Camera"]


class PixelFormat(Enum):
    Mono8 = ext.PIXEL_FORMAT_MONO8
    Mono10p = ext.PIXEL_FORMAT_MONO10p

    @staticmethod
    def maxValue(format):
        if format == PixelFormat.Mono8:
            return 255
        elif format == PixelFormat.Mono10p:
            return 1023


class Camera:
    def __init__(self, pixel_format: PixelFormat = PixelFormat.Mono10p, useCuda: bool = True, fake: bool = False):
        self._camera = ext.Camera(pixel_format.value, useCuda, fake)

    def get_image(self) -> Tuple[torch.Tensor, Tuple[int, float, float]]:
        image, image_info = self._camera.getImage()
        return image, image_info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._camera
        return False

    @property
    def image_ready(self) -> bool:
        return self._camera.imageReady()

    @property
    def exposure(self) -> float:
        return self._camera.getFloatAttribute("ExposureTime")

    @exposure.setter
    def exposure(self, value: float):
        self._camera.setFloatAttribute("ExposureTime", float(value))

    @property
    def gain(self) -> float:
        return self._camera.getFloatAttribute("Gain")

    @gain.setter
    def gain(self, value: float):
        self._camera.setFloatAttribute("Gain", float(value))

    @property
    def analog_gain(self) -> int:
        return self._camera.getIntAttribute("AnalogGain")

    @analog_gain.setter
    def analog_gain(self, value: int):
        self._camera.setIntAttribute("AnalogGain", int(value))

    @property
    def black_level(self) -> float:
        return self._camera.getIntAttribute("BlackLevel")

    @black_level.setter
    def black_level(self, value: int):
        self._camera.setIntAttribute("BlackLevel", int(value))

    @property
    def temperature(self) -> float:
        return self._camera.getFloatAttribute("DeviceTemperature")

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def width(self) -> int:
        return int(self._camera.getIntAttribute("Width"))

    @property
    def height(self) -> int:
        return int(self._camera.getIntAttribute("Height"))

# TorchBeam
TorchBeam aims to provide high performance GPU video streaming with an intuative user interface and an emphesis on scientific computing.
The package can be used to capture video feeds, encode and decode video files, and display video in a GUI all while keeping data strictly on the GPU where possible.

## Support
Currently, TorchBeam is a proof of concept and has very limited support in a number of areas and <ins>very likely does nto support your usecase</ins>. Contributions to expand support are welcome.

## Installation
Due to dependency issues with cuda and pytorch, installation currently requires compiling from source.
Firstly install the system dependencies...
```
# Visualisation depends on OpenGL...
sudo apt install -y libglib2.0-dev libgl-dev
# Video recording depends on ffmpeg...
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
# And camera interfacing is performed with aravis...
sudo apt install -y aravis-tools libaravis-dev libxml2 libxml2-dev libusb-1.0-0-dev
```
Then install the cuda toolkit for compilation...
```
sudo apt install -y nvidia-cuda-toolkit
```
In order to ensure compatability, the NVCC cuda compilation major version must match the version used to build your PyTorch installation.
```
nvcc --version | grep "release" | sed 's/.*release \(.*\),.*/\1/'
python -c "import torch; print(torch.version.cuda)"
```
Note: only the major version is important, e.g. 12.0 and 12.6 are compatable.
Installing a version of PyTorch that matches your nvcc version can be done by specifiying the corresponding index url in the pip install command.
For example, the following command will install pytorch compiled with cuda 12.1...
```
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

Once your environment is setup, you are ready to install torchbeam. This must be done with the --no-build-isolation flag to ensure the compiled code is compatible with your pytorch installation.
```
pip install --no-build-isolation git+https://github.com/charliebudd/torchbeam.git
```


## Usage Examples
This example demonstrates how to stream video footage from a camera and encoding it into a video file
```Python
from torchbeam.aquire import Camera
from torchbeam.record import VideoWriter

camera = Camera()
video = VideoWriter("my-video.mp4", camera.frame_size, 30.0)

while True:
    frame = camera.get_image()
    if frame == None:
        break
    video.write_frame(frame)
```

This example shows a minimal kivy app displaying a video file
```Python
from kivy.app import App
from kivy.clock import Clock
from torchbeam.record import VideoReader
from torchbeam.display.kivy import ImageViewer

class MyApp(App):
    def build(self):
        self.video_reader = VideoReader("my-video.mp4")
        self.image_viewer = ImageViewer(self.video_reader.frame_size)
        Clock.schedule_interval(self.update, 1.0/self.video_reader.fps)
        return self.image_viewer

    def update(self, dt):
        frame = self.video_reader.get_frame()
        self.image_viewer.update_image(frame)

if __name__ == '__main__':
    MyApp().run()
```

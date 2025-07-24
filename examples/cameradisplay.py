from kivy.app import App
from kivy.clock import Clock
from torchbeam.aquire import Camera, PixelFormat
from torchbeam.display.kivy import ImageViewer

class CameraDisplayApp(App):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def build(self):
        self.image_viewer = ImageViewer(self.camera.frame_size)
        Clock.schedule_interval(self.update, 1.0/60.0)
        return self.image_viewer

    def update(self, dt):
        frame, _ = self.camera.get_image()
        self.image_viewer.update_image(frame)

if __name__ == '__main__':
    with Camera(PixelFormat.Mono8) as camera:
        CameraDisplayApp(camera).run()

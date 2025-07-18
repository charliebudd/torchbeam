import torch
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture

import torchbeam_ext as ext


class ImageViewer(BoxLayout):
    def __init__(self, image_size, **kwargs):
        super(ImageViewer, self).__init__(**kwargs)
        # The colorfmt of the texture must be RGBA as CUDA->OpenGL interop does not support RGB
        self.texture = Texture.create(size=image_size[::-1], colorfmt="rgba")
        self.image_widget = Image(texture=self.texture)
        self.image_widget.nocache = True
        self.torchTexture = None
        self.add_widget(self.image_widget)

    def update_image(self, image: torch.Tensor):

        # Grayscale to rgb
        if image.ndim == 2:
            image.unsqueeze(0)
        if image.size(0) == 1:
            image = torch.stack(3 * [image])

        # Formatting tensor for OpenGL texture
        if image.is_floating_point():
            image = (image * 255).to(torch.uint8)
        if image.dtype == torch.int16:
            image = (image / 4).to(torch.uint8)
        image = torch.cat([image, 255 * torch.ones_like(image[:1])], dim=0)
        image = image.permute(1, 2, 0).flip(0)  # Channel last, y+ is up

        # Performing blit and asking for the widget to be redrawn
        if image.is_cuda:

            # Lazy init the torch texture as we may have constructed
            # this GUI element before kivy initialises OpenGL properly
            if self.torchTexture is None:
                if self.texture.id == 0:
                    return  # kivy may still have not initialised fully
                self.torchTexture = ext.TorchTexture(self.texture.id)

            self.torchTexture.blitTexture(image)

        else:
            self.texture.blit_buffer(image.cpu().numpy().tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.image_widget.canvas.ask_update()

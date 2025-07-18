import torch
import torchbeam_ext as ext


__all__ = ["VideoReader"]


class VideoReader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.open = True
        self.video_decoder = ext.TorchVideoDecoder(self.file_path, "gray10le")

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.video_decoder.frameSize()

    @property
    def metadata(self) -> str:
        return self.video_decoder.getMetadata()
    
    def get_frame(self) -> torch.Tensor:
        frame, metadata = self.video_decoder.readFrame()
        if frame.numel() == 0:
            frame = None
        return frame, metadata

    def close(self) -> None:
        if self.open:
            self.video_decoder.close()
            self.open = False

    def __del__(self) -> None:
        self.close()

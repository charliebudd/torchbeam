import torch
import torchbeam_ext as ext
from typing import Tuple

__all__ = ["VideoWriter"]


class VideoWriter:
    def __init__(
        self, file_path: str, frame_size: Tuple[int, int], fps: float, meta_data: str = None, use_cuda: bool = True
    ) -> None:
        self.file_path = file_path
        self.frame_size = frame_size
        self.fps = fps
        self.use_cuda = use_cuda
        self.open = True
        self.video_encoder = ext.TorchVideoEncoder(self.file_path, *self.frame_size, self.fps, "gray10le", "p010le", meta_data)

    def write_frame(self, frame: torch.Tensor, timestamp: int = None, metadata=None) -> None:
        self.video_encoder.writeFrame(frame, timestamp, metadata)

    def close(self) -> None:
        if self.open:
            self.video_encoder.close()
            self.open = False

    def __del__(self) -> None:
        self.close()

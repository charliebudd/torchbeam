import unittest
from parameterized import parameterized_class

from itertools import product
from time import time_ns
import torch
import os

from torchbeam.record import VideoWriter
from torchbeam.record import VideoReader

FILE_PATH = "unittest_video.mp4"

# devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
devices = ["cuda"]  # Not currently implemented for cpu
params = product(devices)


@parameterized_class(("device",), params)
class TestVideoRecording(unittest.TestCase):

    def test_write_read(self):
        fps = 20
        frame_count = 20
        frame_size = (480, 640)

        input_video_metadata = "This is the metadata for the video"
        input_frame_metadata = [f"This is the metadata for frame {i}" for i in range(frame_count)]
        input_frames = (1023 * torch.rand((frame_count, *frame_size), device=self.device)).to(torch.int16)

        if os.path.exists(FILE_PATH):
            os.remove(FILE_PATH)

        video_writer = VideoWriter(FILE_PATH, frame_size, fps, input_video_metadata, use_cuda=input_frames.is_cuda)
        for metadata, frame in zip(input_frame_metadata, input_frames):
            video_writer.write_frame(frame, time_ns(), metadata)
        video_writer.close()

        video_reader = VideoReader(FILE_PATH)
        output_video_metadata = video_reader.metadata
        output_frame_metadata = []
        output_frames = []
        while True:
            image, metadata = video_reader.getFrame()
            if image is None:
                break
            output_frame_metadata.append(metadata)
            output_frames.append(image.to(self.device))
        output_frames = torch.stack(output_frames)

        if os.path.exists(FILE_PATH):
            os.remove(FILE_PATH)

        self.assertEqual(input_video_metadata, output_video_metadata, msg="Video metadata is incorrect")
        self.assertEqual(input_frame_metadata, output_frame_metadata, msg="Frame metadata is incorrect")
        torch.testing.assert_close(input_frames, output_frames, atol=1, rtol=0)

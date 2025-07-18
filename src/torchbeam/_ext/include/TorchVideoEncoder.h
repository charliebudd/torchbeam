#pragma once
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

class TorchVideoEncoder {

public:

    TorchVideoEncoder(std::string filePath, int width, int height, int fps, std::string inPixFmt, std::string outPixFmt, std::string metadata);
    ~TorchVideoEncoder();

    void writeFrame(torch::Tensor frame, uint64_t timestamp, std::string metadata);
    void close();

private:

    void receivePackets();

private:

    AVBufferRef* m_hwDeviceContextBuffer;
    AVBufferRef* m_hwFramesContextBuffer;

    AVFormatContext* m_formatContext;

    AVCodecContext* m_codecContext;
    AVStream* m_stream;

    AVCodecContext* m_metadataCodecContext;
    AVStream* m_metadataStream;

    AVFrame* m_frame;
    AVPacket* m_packet;
    AVPacket* m_metadataPacket;
    int64_t m_firstTimestamp;
};

#pragma once
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

class TorchVideoDecoder {

public:

    TorchVideoDecoder(std::string filePath, std::string outPixFmt);
    ~TorchVideoDecoder();

    std::string getMetadata();
    std::pair<int, int> frameSize();
    std::pair<torch::Tensor, std::string> readFrame();
    void close();

private:

    AVCodecContext* m_codecContext;

    AVFormatContext* m_formatContext;
    AVStream* m_stream;
    int m_videoStreamIndex;
    bool m_noMorePackets;

    AVFrame* m_frame;
    AVPacket* m_packet;
    AVPacket* m_metadataPacket;
};

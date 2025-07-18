#include "../include/TorchVideoDecoder.h"

TorchVideoDecoder::TorchVideoDecoder(std::string filepath, std::string outputPixFmt) {

    // Open video file
    m_formatContext = avformat_alloc_context();
    int ret = avformat_open_input(&m_formatContext, filepath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Failed to open video");
    }

    // Find video stream info
    ret = avformat_find_stream_info(m_formatContext, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Failed to find stream");
    }

    // Find decoder for the video stream
    const AVCodec* codec;
    m_videoStreamIndex = av_find_best_stream(m_formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (m_videoStreamIndex < 0) {
        throw std::runtime_error("Failed to find best stream");
    }
    m_stream = m_formatContext->streams[m_videoStreamIndex];

    m_codecContext = avcodec_alloc_context3(codec);
    if (!m_codecContext) {
        throw std::runtime_error("Failed create codec context");
    }

    ret = avcodec_parameters_to_context(m_codecContext, m_stream->codecpar);
    if (ret < 0) {
        throw std::runtime_error("Failed to find copy codec parameters from stream");
    }

    m_codecContext->framerate = av_guess_frame_rate(m_formatContext, m_stream, NULL);
    m_codecContext->flags |= AV_CODEC_FLAG_COPY_OPAQUE;

    ret = avcodec_open2(m_codecContext, codec, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Failed open codec");
    }

    m_packet = av_packet_alloc();
    m_metadataPacket = av_packet_alloc();
    m_frame = av_frame_alloc();

    m_noMorePackets = false;
}

TorchVideoDecoder::~TorchVideoDecoder() {
}

std::string TorchVideoDecoder::getMetadata() {
    AVDictionaryEntry* entry = av_dict_get(m_formatContext->metadata, "torchbeam_metadata", NULL, 0);
    return std::string(entry->value);
}
std::pair<torch::Tensor, std::string> TorchVideoDecoder::readFrame() {

    int ret = 0;

    while (true) {

        if (!m_noMorePackets) {
            ret = av_read_frame(m_formatContext, m_packet);
            if (ret < 0) {
                m_noMorePackets = true;
            }
            else {
                
                if (m_packet->stream_index == 0) {
                    
                    // This is a bit brittle but next packet "should" be the metadata...
                    ret = av_read_frame(m_formatContext, m_metadataPacket);
                    if (ret < 0) {
                        throw std::runtime_error("Failed read frame");
                    }
                    m_packet->opaque_ref = av_buffer_alloc(m_metadataPacket->size);
                    std::memcpy(m_packet->opaque_ref->data, m_metadataPacket->data, m_metadataPacket->size);

                    ret = avcodec_send_packet(m_codecContext, m_packet);
                    if (ret < 0) {
                        throw std::runtime_error("Failed to send packet");
                    }
                }
            }
        }

        ret = avcodec_receive_frame(m_codecContext, m_frame);

        if (ret == AVERROR(EAGAIN)) {
            if (m_noMorePackets) {
                ret = avcodec_send_packet(m_codecContext, NULL);
                if (ret < 0) {
                    throw std::runtime_error("Failed to send packet");
                }
            }
            continue;
        }
        else if (ret == 0) {
            torch::Tensor result = torch::from_blob(m_frame->data[0], {m_frame->height, m_frame->width}, torch::dtype(torch::kInt16));
            return std::pair(result, (char*)m_frame->opaque_ref->data);
        }
        else if (ret == AVERROR_EOF) {
            return std::pair(torch::empty({0}), "");
        }
        else {
            throw std::runtime_error("Failed to receive frame");
        }
    }
}

void TorchVideoDecoder::close() {
    av_packet_free(&m_packet);
    av_packet_free(&m_metadataPacket);
    av_frame_free(&m_frame);
    avcodec_close(m_codecContext);
    avformat_close_input(&m_formatContext);
    avformat_free_context(m_formatContext);
}


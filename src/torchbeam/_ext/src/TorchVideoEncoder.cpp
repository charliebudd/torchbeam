#include "../include/TorchVideoEncoder.h"

#include <libavutil/error.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../include/CudaConvert.h"

TorchVideoEncoder::TorchVideoEncoder(std::string filename, int height, int width, int fps, std::string inPixFmt,
                                     std::string outPixFmt, std::string metadata) {
                                        
    AVPixelFormat outputPixFmt = av_get_pix_fmt(outPixFmt.c_str());

    av_log_set_level(AV_LOG_QUIET);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int ret = av_hwdevice_ctx_create(&m_hwDeviceContextBuffer, AV_HWDEVICE_TYPE_CUDA, prop.name, NULL, 0);
    if (ret < 0) {
        throw std::runtime_error("Could not open hw device");
    }

    m_hwFramesContextBuffer = av_hwframe_ctx_alloc(m_hwDeviceContextBuffer);

    AVHWFramesContext* frameCtxPtr = (AVHWFramesContext*)(m_hwFramesContextBuffer->data);
    frameCtxPtr->width = width;
    frameCtxPtr->height = height;
    frameCtxPtr->sw_format = outputPixFmt;
    frameCtxPtr->format = AV_PIX_FMT_CUDA;

    ret = av_hwframe_ctx_init(m_hwFramesContextBuffer);
    if (ret < 0) {
        throw std::runtime_error("Failed to allocate cuda frame");
    }

    // ######################
    // Codec stuff...

    const AVCodec* codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) {
        throw std::runtime_error("hevc_nvenc codec not found");
    }

    m_codecContext = avcodec_alloc_context3(codec);
    if (!m_codecContext) {
        throw std::runtime_error("Can't allocate video codec context");
    }

    m_codecContext->height = height;
    m_codecContext->width = width;
    m_codecContext->time_base = {1, int(1e9)};  // Nanoseconds
    m_codecContext->framerate = {0, 1};         // Infered from pts
    m_codecContext->hw_device_ctx = av_buffer_ref(m_hwDeviceContextBuffer);
    m_codecContext->hw_frames_ctx = av_buffer_ref(m_hwFramesContextBuffer);
    m_codecContext->codec_type = AVMEDIA_TYPE_VIDEO;
    m_codecContext->pix_fmt = AV_PIX_FMT_CUDA;
    m_codecContext->sw_pix_fmt = outputPixFmt;
    m_codecContext->color_range = AVCOL_RANGE_JPEG;
    m_codecContext->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
    m_codecContext->max_b_frames = 0; // Disable B-frames

    ret = av_opt_set(m_codecContext->priv_data, "tune", "lossless", 0);
    if (ret < 0) {
        throw std::runtime_error("Could not set lossless");
    }

    ret = avcodec_open2(m_codecContext, codec, nullptr);
    if (ret < 0) {
        throw std::runtime_error("Could not open codec");
    }

    // ##########

    const AVCodec* metadataCodec = avcodec_find_encoder_by_name("mov_text");
    if (!metadataCodec) {
        throw std::runtime_error("Subtitle codec not found");
    }

    // Allocate codec context
    m_metadataCodecContext = avcodec_alloc_context3(metadataCodec);
    if (!m_metadataCodecContext) {
        throw std::runtime_error("Can't allocate subtitle codec context");
    }

    // Set codec parameters
    m_metadataCodecContext->codec_type = AVMEDIA_TYPE_SUBTITLE;
    m_metadataCodecContext->codec_id = AV_CODEC_ID_MOV_TEXT;
    m_metadataCodecContext->time_base = m_codecContext->time_base;
    m_metadataCodecContext->framerate = m_codecContext->framerate;

    // ######################
    // Muxing stuff...

    m_formatContext = avformat_alloc_context();
    if (!m_formatContext) {
        throw std::runtime_error("Failed to allocate format context");
    }

    const AVOutputFormat* outputFormat = av_guess_format(NULL, filename.c_str(), NULL);
    m_formatContext->oformat = outputFormat;

    m_stream = avformat_new_stream(m_formatContext, codec);
    avcodec_parameters_from_context(m_stream->codecpar, m_codecContext);
    m_stream->time_base = m_codecContext->time_base;
    m_stream->id = 0;

    m_metadataStream = avformat_new_stream(m_formatContext, nullptr);
    avcodec_parameters_from_context(m_metadataStream->codecpar, m_metadataCodecContext);
    m_metadataStream->time_base = m_codecContext->time_base;
    m_metadataStream->id = 1;

    AVDictionary* formatOptions = NULL;
    av_dict_set(&formatOptions, "brand", "mp42", 0);
    av_dict_set(&formatOptions, "movflags", "use_metadata_tags", 0);
    av_dict_set(&m_formatContext->metadata, "torchbeam_metadata", metadata.c_str(), 0);

    ret = avio_open(&m_formatContext->pb, filename.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        throw std::runtime_error("Failed to open file");
    }

    ret = avformat_write_header(m_formatContext, &formatOptions);
    if (ret < 0) {
        throw std::runtime_error("Failed to write header");
    }

    // ######################
    // Frame/packet stuff...

    m_frame = av_frame_alloc();
    if (!m_frame) {
        throw std::runtime_error("Could not allocate video frame");
    }
    m_frame->format = m_codecContext->pix_fmt;
    m_frame->height = m_codecContext->height;
    m_frame->width = m_codecContext->width;

    ret = av_hwframe_get_buffer(m_hwFramesContextBuffer, m_frame, 0);
    if (ret < 0) {
        throw std::runtime_error("Failed to allocate cuda frame");
    }

    // Initialising UV planes to 0.5 for grayscale
    av_frame_make_writable(m_frame);
    uint16_t* tempBuffer = (uint16_t*)std::malloc(2 * width * height / 2);
    for (int i = 0; i < width * height / 2; i++) {
        tempBuffer[i] = 32768;
    }
    cudaMemcpy2D(m_frame->data[1], m_frame->linesize[1], tempBuffer, 2 * width, 2 * width, height / 2, cudaMemcpyHostToDevice);
    free(tempBuffer);

    m_packet = av_packet_alloc();
    if (!m_packet) {
        throw std::runtime_error("Could not allocate packet");
    }

    m_metadataPacket = av_packet_alloc();
    if (!m_packet) {
        throw std::runtime_error("Could not allocate packet");
    }

    m_firstTimestamp = 0;
}

void TorchVideoEncoder::writeFrame(torch::Tensor frame, uint64_t timestamp, std::string metadata) {
    cudaDeviceSynchronize();
    av_frame_make_writable(m_frame);
    frame = frame.contiguous();
    gray10le_to_p010le((uint16_t*)m_frame->data[0], m_frame->linesize[0] / 2, (uint16_t*)frame.data_ptr(),
                       m_codecContext->width, m_codecContext->width, m_codecContext->height);
    cudaDeviceSynchronize();

    if (m_firstTimestamp == 0) {
        m_firstTimestamp = timestamp;
    }
    m_frame->pts = timestamp - m_firstTimestamp;

    m_frame->opaque_ref = av_buffer_alloc(metadata.length() + 1);
    std::memcpy(m_frame->opaque_ref->data, metadata.c_str(), metadata.length() + 1);

    int ret = avcodec_send_frame(m_codecContext, m_frame);
    m_frame->pts = 42;
    if (ret < 0) {
        throw std::runtime_error("Failed to send frame");
    }
    receivePackets();
}

void TorchVideoEncoder::close() {
    avcodec_send_frame(m_codecContext, NULL);
    receivePackets();
    av_interleaved_write_frame(m_formatContext, NULL);

    av_write_trailer(m_formatContext);
    avio_close(m_formatContext->pb);
}

void TorchVideoEncoder::receivePackets() {
    while (true) {
        int ret = avcodec_receive_packet(m_codecContext, m_packet);
        if (ret == 0) {
            av_packet_rescale_ts(m_packet, m_codecContext->time_base, m_stream->time_base);

            m_metadataPacket->data = m_packet->opaque_ref->data;
            m_metadataPacket->size = m_packet->opaque_ref->size;
            m_metadataPacket->pts = m_packet->pts;
            m_metadataPacket->dts = m_packet->dts;
            m_metadataPacket->duration = m_packet->duration;
            m_metadataPacket->stream_index = m_metadataStream->index;
            ret = av_interleaved_write_frame(m_formatContext, m_metadataPacket);
            if (ret < 0) {
                throw std::runtime_error("Failed to write metadata packet");
            }

            m_packet->stream_index = m_stream->index;
            ret = av_interleaved_write_frame(m_formatContext, m_packet);
            if (ret < 0) {
                throw std::runtime_error("Failed to write packet");
            }

            av_packet_unref(m_packet);

            continue;
        } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_unref(m_packet);
            break;
        } else {
            throw std::runtime_error("Encoder error");
        }
    }
}

TorchVideoEncoder::~TorchVideoEncoder() {
    avformat_free_context(m_formatContext);
    av_frame_free(&m_frame);
    av_packet_free(&m_packet);
    av_packet_free(&m_metadataPacket);
    avcodec_free_context(&m_codecContext);
    av_buffer_unref(&m_hwFramesContextBuffer);
    av_buffer_unref(&m_hwDeviceContextBuffer);
}

#pragma once 

enum class PixelFormat {
    Mono8 = 0,
    Mono10p = 1
};

constexpr const char* PF_TO_STRINGS[3] = {
    "Mono8",
    "Mono10p"
};

constexpr int PF_TO_BITS[3] = {8, 10};

constexpr bool pixelFormatIs8Bit(PixelFormat pixelFormat) {
    return pixelFormat == PixelFormat::Mono8;
}

constexpr int pixelFormatBitCount(PixelFormat pixelFormat) {
    return PF_TO_BITS[static_cast<int>(pixelFormat)];
}

constexpr const char* pixelFormatName(PixelFormat pixelFormat) {
    return PF_TO_STRINGS[static_cast<int>(pixelFormat)];
}

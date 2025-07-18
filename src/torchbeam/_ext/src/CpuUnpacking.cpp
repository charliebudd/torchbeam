#include <cstring>
#include "../include/Unpacking.h"

// We can definetly improve this with some vectorisation,
// but we will probably use the GPU for this.
void cpuUnpack10to16(uint16_t* dest, const uint8_t* src, int intCount) {
    uint64_t val;
    int mask = 0x3FF;
    const uint8_t* max = src + (intCount * 10 / 8);
    for (const uint8_t *ptr = src; ptr < max; ptr += 5) {
        std::memcpy(&val, ptr, sizeof(val));
        *dest++ = (val >>  0) & mask;
        *dest++ = (val >> 10) & mask;
        *dest++ = (val >> 20) & mask;
        *dest++ = (val >> 30) & mask;
    }
}
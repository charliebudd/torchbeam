#pragma once 

#include <stdint.h>

void gray10le_to_p010le(uint16_t* dst, int dstPitch, const uint16_t* src, int srcPitch, int srcWidth, int srcHeight);

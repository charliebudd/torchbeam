#pragma once 

#include <stdint.h>

void cpuUnpack10to16(uint16_t* dest, const uint8_t* src, int intCount);
void cudaUnpack10to16(void* dest, void* src, int intCount);

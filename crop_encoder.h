#pragma once

#include <cstdint>
#include <vector>

struct decoded_frame;

// Encodes a BGR24 frame to JPEG in memory. Returns an empty vector on failure — encoding is
// always best-effort and must never abort the detection itself. video_a keeps no crop files:
// the JPEG travels in DetectionEvent.crop_jpeg and persisting it is the caller's concern.
std::vector<uint8_t> encode_crop_jpeg(const decoded_frame& crop);

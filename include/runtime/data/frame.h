#pragma once

#include <cstdint>

#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace ptk::data
{
  struct Frame
  {
    TensorView image;               // image tensor
    core::PixelFormat pixel_format; // channel interpretation
    core::TensorLayout layout;
    int64_t timestamp_ns; // timestamp in from context
    int64_t frame_index;  // optional sequential index
    int camera_id;        // optional identifier

    Frame()
        : image(),
          pixel_format(core::PixelFormat::kUnknown),
          layout(core::TensorLayout::kUnknown),
          timestamp_ns(0),
          frame_index(0),
          camera_id(0) {}
  };
} // namespace ptk::data
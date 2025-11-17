#ifndef RUNTIME_DATA_FRAME_H_
#define RUNTIME_DATA_FRAME_H_

#include <cstdint>

#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

namespace runtime {

struct Frame {
  TensorView image; //image tensor
  PixelFormat pixel_format; //channel interpretation
  int64_t timestamp_ns; //timestamp in from context
  int64_t frame_index; //optional sequential index
  int camera_id; //optional identifier

  Frame()
    : image(),
        pixel_format(PixelFormat::kUnknown),
        timestamp_ns(0),
        frame_index(0),
        camera_id(0) {}
};

}  // namespace runtime

#endif  // RUNTIME_DATA_FRAME_H_
#pragma once

#include <cstdint>
#include <vector>
#include <memory>

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

    std::shared_ptr<std::vector<uint8_t>> owned_data;

    Frame()
        : image(),
          pixel_format(core::PixelFormat::kUnknown),
          layout(core::TensorLayout::kUnknown),
          timestamp_ns(0),
          frame_index(0),
          camera_id(0),
          owned_data(nullptr) {}

    static Frame CreateOwned(int height, int width, int channels, 
                            core::PixelFormat format, core::TensorLayout layout_type)
    {
        Frame frame;
        frame.owned_data = std::make_shared<std::vector<uint8_t>>(height * width * channels);
        frame.image = TensorView(
            BufferView(frame.owned_data->data(), frame.owned_data->size(), core::DeviceType::kCpu),
            core::DataType::kUint8,
            TensorShape({static_cast<int64_t>(height), static_cast<int64_t>(width), static_cast<int64_t>(channels)})
        );
        frame.pixel_format = format;
        frame.layout = layout_type;
        return frame;
    }
  };
} // namespace ptk::data
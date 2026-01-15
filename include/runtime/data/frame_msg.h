#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"
#include "runtime/data/frame.h"

namespace ptk::data {

// Zero-copy Frame message that owns its data
struct FrameMsg {
    // Owned buffer data (for zero-copy transfer)
    std::vector<uint8_t> buffer_data;
    
    // Tensor metadata
    TensorShape shape;
    core::DataType dtype;
    
    // Frame metadata
    core::PixelFormat pixel_format;
    core::TensorLayout layout;
    int64_t timestamp_ns;
    int64_t frame_index;
    int camera_id;
    
    // Factory method to create from owned data
    static std::unique_ptr<FrameMsg> Create(
        std::vector<uint8_t>&& data,
        core::DataType dtype,
        const TensorShape& shape,
        core::PixelFormat pixel_format,
        core::TensorLayout layout,
        int64_t timestamp_ns,
        int64_t frame_index,
        int camera_id)
    {
        auto msg = std::make_unique<FrameMsg>();
        msg->buffer_data = std::move(data);
        msg->dtype = dtype;
        msg->shape = shape;
        msg->pixel_format = pixel_format;
        msg->layout = layout;
        msg->timestamp_ns = timestamp_ns;
        msg->frame_index = frame_index;
        msg->camera_id = camera_id;
        return msg;
    }
    
    // Get a TensorView pointing to owned data (non-owning view)
    TensorView GetTensorView() const {
        BufferView buffer_view(
            const_cast<uint8_t*>(buffer_data.data()),
            buffer_data.size(),
            core::DeviceType::kCpu);
        return TensorView(buffer_view, dtype, shape);
    }
    
    // Convert to Frame (creates non-owning view)
    Frame ToFrame() const {
        Frame frame;
        frame.image = GetTensorView();
        frame.pixel_format = pixel_format;
        frame.layout = layout;
        frame.timestamp_ns = timestamp_ns;
        frame.frame_index = frame_index;
        frame.camera_id = camera_id;
        return frame;
    }
};

} // namespace ptk::data

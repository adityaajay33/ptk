#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"
#include "runtime/data/frame.h"

namespace ptk::data
{

    struct FrameMsg
    {
        std::vector<uint8_t> buffer_data;

        TensorShape shape;
        core::DataType dtype;

        core::PixelFormat pixel_format;
        core::TensorLayout layout;
        int64_t timestamp_ns;
        int64_t frame_index;
        int camera_id;

        static std::unique_ptr<FrameMsg> Create(
            std::vector<uint8_t> &&data,
            core::DataType dtype,
            const TensorShape &shape,
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

        TensorView GetTensorView() const
        {
            BufferView buffer_view(
                const_cast<uint8_t *>(buffer_data.data()),
                buffer_data.size(),
                core::DeviceType::kCpu);
            return TensorView(buffer_view, dtype, shape);
        }

        Frame ToFrame() const
        {
            Frame frame;
            frame.image = GetTensorView();
            frame.pixel_format = pixel_format;
            frame.layout = layout;
            frame.timestamp_ns = timestamp_ns;
            frame.frame_index = frame_index;
            frame.camera_id = camera_id;
            return frame;
        }

        struct StageTimestamp
        {
            std::string stage_name;
            int64_t timestamp_ns;
        };
        std::vector<StageTimestamp> stage_timestamps;

        void RecordStageEntry(const std::string &stage_name)
        {
            stage_timestamps.push_back({stage_name, CurrentTimeNs()});
        }

        int64_t GetFrameAgeNs() const
        {
            if (stage_timestamps.empty())
                return 0;
            return stage_timestamps.back().timestamp_ns - timestamp_ns;
        }

    private:
        static int64_t CurrentTimeNs()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }
    };

} // namespace ptk::data
#pragma once

#include "sensors/camera_interface.h"
#include "runtime/core/status.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include <vector>
#include <cstdint>

namespace ptk::sensors
{

        class MacCamera : public CameraInterface
        {
        public:
            explicit MacCamera(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
            virtual ~MacCamera();

            core::Status Init() override;
            core::Status Start() override;
            core::Status Stop() override;

            core::Status GetFrame(ptk::data::Frame *out) override;

            void Tick() override;
            
            // Bind output port for scheduler-driven operation
            void BindOutput(core::OutputPort<data::Frame>* out);

        private:
            int device_index_;
            bool is_running_;
            int frame_index_;

            struct Impl;
            Impl *impl_;
            
            // Persistent buffer for zero-copy frame data
            std::vector<uint8_t> frame_buffer_;
            core::OutputPort<data::Frame>* output_;
        };

} // namespace ptk::sensors
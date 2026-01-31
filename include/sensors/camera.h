#pragma once

#include "sensors/camera_interface.h"
#include "runtime/core/status.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include <vector>
#include <cstdint>

namespace ptk::sensors
{

        class Camera : public CameraInterface
        {
        public:
            explicit Camera(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
            virtual ~Camera();

            core::Status Init() override;
            core::Status Start() override;
            core::Status Stop() override;

            core::Status GetFrame(ptk::data::Frame *out) override;

            void Tick() override;

            void BindOutput(core::OutputPort<data::Frame>* out);

        private:
            int device_index_;
            bool is_running_;
            int frame_index_;

            struct Impl;
            Impl *impl_;

            core::OutputPort<data::Frame>* output_;
        };

} // namespace ptk::sensors

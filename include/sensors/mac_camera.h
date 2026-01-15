#pragma once

#include "sensors/camera_interface.h"
#include "runtime/core/status.h"
#include "runtime/data/frame.h"
#include "runtime/data/frame_msg.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk::sensors
{

        class MacCamera : public CameraInterface
        {
        public:
            explicit MacCamera(
                int device_index,
                const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
            virtual ~MacCamera();

            core::Status Init() override;
            core::Status Start() override;
            core::Status Stop() override;

            core::Status GetFrame(ptk::data::Frame *out) override;

            void Tick() override;

        private:
            void PublishFrame();
            
            int device_index_;
            bool is_running_;
            int frame_index_;

            struct Impl;
            Impl *impl_;
            
            rclcpp::Publisher<data::FrameMsg>::SharedPtr frame_publisher_;
        };

} // namespace ptk::sensors
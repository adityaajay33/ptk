#pragma once

#include "runtime/core/status.h"
#include "runtime/data/frame.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk::sensors
{

        class CameraInterface : public rclcpp::Node
        {
        public:
            explicit CameraInterface(
                const std::string &node_name,
                const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
                : rclcpp::Node(node_name, options) {}

            virtual ~CameraInterface() {}

            virtual core::Status Init() = 0;
            virtual core::Status Start() = 0;
            virtual core::Status Stop() = 0;

            virtual core::Status GetFrame(ptk::data::Frame *out) = 0;

            virtual void Tick() = 0;
        };

} // namespace ptk::sensors
// include/runtime/components/frame_debugger.h
#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "runtime/data/frame_msg.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk::components
{    class FrameDebugger : public ComponentInterface
    {
    public:
      explicit FrameDebugger(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
      ~FrameDebugger() override = default;

      // The pipeline or app calls this to connect a Frame source.
      void BindInput(core::InputPort<data::Frame> *port);

      core::Status Init(core::RuntimeContext *context) override;
      core::Status Start() override;
      core::Status Stop() override;
      void Tick() override;

    private:
      void FrameCallback(std::unique_ptr<data::FrameMsg> msg);
      
      core::RuntimeContext *context_;
      core::InputPort<data::Frame> *input_;
      int tick_count_;
      
      rclcpp::Subscription<data::FrameMsg>::SharedPtr frame_subscription_;
    };

}  // namespace ptk::components
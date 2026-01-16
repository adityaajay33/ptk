// include/runtime/components/frame_debugger.h
#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"

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
      core::RuntimeContext *context_;
      core::InputPort<data::Frame> *input_;
      int tick_count_;
    };

}  // namespace ptk::components
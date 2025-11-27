#pragma once

#include "runtime/components/component_interface.h"

namespace ptk::components
{    class Heartbeat : public ComponentInterface
    {
    public:
      Heartbeat();
      ~Heartbeat() override = default;

      core::Status Init(core::RuntimeContext *context) override;
      core::Status Start() override;
      core::Status Stop() override;
      void Tick() override;

    private:
      core::RuntimeContext *context_;
      int count_;
    };

}  // namespace ptk::components
#pragma once

#include "runtime/components/component_interface.h"

namespace ptk {
namespace components {

    class Counter : public ComponentInterface {
        public:
            Counter();
            ~Counter() override = default;

            core::Status Init(core::RuntimeContext* context) override;
            core::Status Start() override;
            core::Status Stop() override;
            void Tick() override;

        private:
            core::RuntimeContext* context_;
            int count_;
    };

} // namespace components
} // namespace ptk
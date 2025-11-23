#ifndef RUNTIME_COMPONENTS_COUNTER_H_
#define RUNTIME_COMPONENTS_COUNTER_H_

#include "runtime/components/component_interface.h"

namespace ptk {
namespace components {

    class Counter : public ComponentInterface {
        public:
            Counter();
            ~Counter() override = default;

            core::Status Init(core::RuntimeContext* context) override;
            core::Status Start() override;
            void Stop() override;
            void Tick() override;

        private:
            core::RuntimeContext* context_;
            int count_;
    };

} // namespace components
} // namespace ptk

#endif // RUNTIME_COMPONENTS_COUNTER_H_
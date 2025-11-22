#ifndef RUNTIME_COMPONENTS_COUNTER_H_
#define RUNTIME_COMPONENTS_COUNTER_H_

#include "runtime/components/component_interface.h"

namespace ptk {

    class Counter : public ComponentInterface {
        public:
            Counter();
            ~Counter() override = default;

            Status Init(RuntimeContext* context) override;
            Status Start() override;
            void Stop() override;
            void Tick() override;

        private:
            RuntimeContext* context_;
            int count_;
    };

} // namespace ptk

#endif // RUNTIME_COMPONENTS_COUNTER_H_
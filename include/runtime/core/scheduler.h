#ifndef RUNTIME_CORE_SCHEDULER_H_
#define RUNTIME_CORE_SCHEDULER_H_


#include <vector>

#include "runtime/components/component_interface.h"
#include "runtime/core/status.h"


namespace ptk {
namespace core {

    class RuntimeContext;

    class Scheduler {

        public:

            Scheduler();

            Status Init(RuntimeContext* context);
            Status AddComponent(components::ComponentInterface* component);
            Status Start();
            void Stop();
            void RunLoop(int num_ticks);

        private:

            RuntimeContext* context_;

            std::vector<components::ComponentInterface*> components_;
            bool running_;
            int tick_;
    };

} // namespace core
}

#endif // RUNTIME_CORE_SCHEDULER_H_
#ifndef RUNTIME_CORE_PIPELINE_H_
#define RUNTIME_CORE_PIPELINE_H_

#include "runtime/core/status.h"
#include "runtime/core/scheduler.h"
#include "runtime/components/component_interface.h"

namespace ptk {
namespace core {

    class RuntimeContext;

    class Pipeline {
        public:
            Pipeline();
            virtual ~Pipeline();

            Pipeline(const Pipeline&) = delete;
            Pipeline& operator=(const Pipeline&) = delete;

            Status Build(RuntimeContext* context);
            Status RegisterComponents(Scheduler* scheduler);

        protected:
            
            virtual Status DoBuild(RuntimeContext* context) = 0;

            Status AddComponent(components::ComponentInterface* component);

        private:
            RuntimeContext* context_;
            Scheduler* scheduler_;
            bool built_;
            std::vector<components::ComponentInterface*> components_;
    };

} // namespace core
} // namespace ptk

#endif // RUNTIME_CORE_PIPELINE_H_
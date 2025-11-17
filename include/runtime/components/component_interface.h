#ifndef RUNTIME_COMPONENTS_COMPONENT_INTERFACE_H_
#define RUNTIME_COMPONENTS_COMPONENT_INTERFACE_H_

#include "runtime/core/status.h"

namespace runtime {

    class RuntimeContext;

    class ComponentInterface {

        public:

            virtual ~ComponentInterface() = default;

            virtual Status Init(RuntimeContext* context) = 0; // called once before start

            virtual Status Start() = 0; // called once before the first ticker

            virtual void Stop() = 0; // called once after the lasttick

            virtual void Tick() = 0; // called repeatedly by scheduler or external driver

    };

} // namespace runtime

#endif // RUNTIME_COMPONENTS_COMPONENT_INTERFACE_H_
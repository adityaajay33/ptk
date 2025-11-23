#ifndef RUNTIME_COMPONENTS_SYNTHETIC_CAMERA_H_
#define RUNTIME_COMPONENTS_SYNTHETIC_CAMERA_H_

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"

namespace ptk {
namespace components {

    class SyntheticCamera : public ComponentInterface {

        public:   
            SyntheticCamera();
            ~SyntheticCamera() override = default;

            // The pipeline or app calls this to connect a Frame sink.
            void BindOutput(core::OutputPort<data::Frame>* port);

            core::Status Init(core::RuntimeContext* context) override;
            core::Status Start() override;
            void Stop() override;
            void Tick() override;

         private:
            core::RuntimeContext* context_;
            core::OutputPort<data::Frame>* output_;
            int frame_index_;
    };

} // namespace components
}

#endif // RUNTIME_COMPONENTS_SYNTHETIC_CAMERA_H_
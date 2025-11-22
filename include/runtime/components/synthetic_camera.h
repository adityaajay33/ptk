#ifndef RUNTIME_COMPONENTS_SYNTHETIC_CAMERA_H_
#define RUNTIME_COMPONENTS_SYNTHETIC_CAMERA_H_

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"

namespace ptk {

    class SyntheticCamera : public ComponentInterface {

        public:   
            SyntheticCamera();
            ~SyntheticCamera() override = default;

            // The pipeline or app calls this to connect a Frame sink.
            void BindOutput(OutputPort<Frame>* port);

            Status Init(RuntimeContext* context) override;
            Status Start() override;
            void Stop() override;
            void Tick() override;

         private:
            RuntimeContext* context_;
            OutputPort<Frame>* output_;
            int frame_index_;
    };
}

#endif // RUNTIME_COMPONENTS_SYNTHETIC_CAMERA_H_
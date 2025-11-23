#ifndef SENSORS_CAMERA_INTERFACE_H
#define SENSORS_CAMERA_INTERFACE_H

#include "runtime/components/component_interface.h"
#include "runtime/data/frame.h"

namespace ptk {
    namespace sensors {
        class CameraInterface : public components::ComponentInterface {
            public:
                virtual ~CameraInterface() {};

                virtual core::Status Init() = 0;
                virtual core::Status Start() = 0;
                virtual core::Status Stop() = 0;

                virtual core::Status GetFrame(data::Frame* out) = 0;
            };
    }
}

#endif // SENSORS_CAMERA_INTERFACE_H
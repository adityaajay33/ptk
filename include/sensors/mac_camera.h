#ifndef SENSORS_MAC_CAMERA_H
#define SENSORS_MAC_CAMERA_H

#include "sensors/camera_interface.h"
#include "runtime/core/types.h"
#include "runtime/core/status.h"
#include "runtime/data/buffer.h"
#include "runtime/data/frame.h"
#include "runtime/core/port.h"

namespace ptk
{
    namespace sensors
    {
        class MacCamera : public sensors::CameraInterface
        {
        public:
            explicit MacCamera(int device_index);
            virtual ~MacCamera();

            core::Status Init() override;
            core::Status Start() override;
            core::Status Stop() = 0;

            core::Status GetFrame(data::Frame *out) override;
            void Tick() override;

            core::Status GetFrame(data::Frame *out) override;

            core::OutputPort<ptk::data::Frame>& output_port() { return output_port_; }

        private:
            int device_index_;
            bool is_running_;
            int frame_index_;

            struct Impl;
            Impl *impl_;

            core::OutputPort<ptk::data::Frame> output_port_;
        };
    }
} // namespace ptk

#endif // SENSORS_MAC_CAMERA_H
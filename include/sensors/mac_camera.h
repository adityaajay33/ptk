#pragma once

#include "sensors/camera_interface.h"
#include "runtime/core/status.h"
#include "runtime/data/frame.h"

namespace ptk
{
    namespace sensors
    {

        class MacCamera : public CameraInterface
        {
        public:
            explicit MacCamera(int device_index);
            virtual ~MacCamera();

            core::Status Init() override;
            core::Status Start() override;
            core::Status Stop() override;

            core::Status GetFrame(ptk::data::Frame *out) override;

            void Tick() override;

        private:
            int device_index_;
            bool is_running_;
            int frame_index_;

            struct Impl;
            Impl *impl_;
        };

    } // namespace sensors
} // namespace ptk
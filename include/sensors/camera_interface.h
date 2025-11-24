#pragma once

#include "runtime/core/status.h"
#include "runtime/data/frame.h"

namespace ptk
{
    namespace sensors
    {

        class CameraInterface
        {
        public:
            virtual ~CameraInterface() {}

            virtual core::Status Init() = 0;
            virtual core::Status Start() = 0;
            virtual core::Status Stop() = 0;

            virtual core::Status GetFrame(ptk::data::Frame *out) = 0;

            virtual void Tick() = 0;
        };

    } // namespace sensors
} // namespace ptk
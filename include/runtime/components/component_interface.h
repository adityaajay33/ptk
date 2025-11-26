#pragma once

#include "runtime/core/status.h"

namespace ptk
{
    namespace core
    {
        class RuntimeContext;
    } // namespace core

    namespace components
    {

        class ComponentInterface
        {

        public:
            virtual ~ComponentInterface() = default;

            virtual core::Status Init(core::RuntimeContext *context) = 0; // called once before start

            virtual core::Status Start() = 0; // called once before the first ticker

            virtual core::Status Stop() = 0; // called once after the lasttick

            virtual void Tick() = 0; // called repeatedly by scheduler or external driver
        };

    } // namespace components
} // namespace ptk
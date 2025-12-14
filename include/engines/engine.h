#pragma once

#include <string>
#include <vector>
#include <memory>

#include "runtime/data/tensor.h"
#include "runtime/core/status.h"
#include "runtime/core/logger.h"
#include "engine_config.h"

namespace ptk::perception {

    class Engine {
        public:
            virtual ~Engine() = default;

            // Load model with Status return
            virtual core::EngineStatus Load(const std::string& model_path) = 0;

            // Run inference with Status return
            virtual core::EngineStatus Infer(const std::vector<data::TensorView>& inputs,
                                             std::vector<data::TensorView>& outputs) = 0;

            virtual std::vector<std::string> InputNames() const = 0;
            virtual std::vector<std::string> OutputNames() const = 0;

            virtual void SetConfig(const EngineConfig& config){
                config_ = config;
            }

        protected:
            EngineConfig config_;
    };
}
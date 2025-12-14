#pragma once

#include "engine.h"
#include "engine_config.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

#include "runtime/data/tensor.h"

namespace ptk::perception {

    class OnnxEngine : public Engine {
        public:
            explicit OnnxEngine(const EngineConfig& config);
            ~OnnxEngine() override;

            core::EngineStatus Load(const std::string& model_path) override;

            core::EngineStatus Infer(const std::vector<data::TensorView>& inputs, std::vector<data::TensorView>& outputs) override;

            std::vector<std::string> InputNames() const override { return input_names_; }
            std::vector<std::string> OutputNames() const override { return output_names_; }

        private:
            Ort::Env env_;
            std::unique_ptr<Ort::Session> session_;
            Ort::SessionOptions session_options_;
            std::vector<std::string> input_names_;
            std::vector<std::string> output_names_;
            EngineConfig config_;
            OnnxRuntimeExecutionProvider active_provider_;

            core::EngineStatus InitializeExecutionProviders();
            bool TryInitializeCudaProvider();
            bool TryInitializeTensorRtProvider();
            bool TryInitializeCpuProvider();
            core::EngineStatus InitializeProvider(OnnxRuntimeExecutionProvider provider);

            Ort::Value CreateOrtTensorFromPtk(const data::TensorView& tv);
    };
}
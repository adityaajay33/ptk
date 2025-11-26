#pragma once

#include "engine.h"
#include <onnxruntime_cxx_api.h>

namespace ptk{

    namespace perception{
        class OnxxEngine : public Engine {
            public:
                OnxxEngine();
                ~OnxxEngine() override;

                bool Load(const std::string& model_path) override;

                bool Infer(const std::vector<EngineTensor>& inputs,std::vector<EngineTensor>* outputs) override;

                std::vector<std::string> InputNames() const override { return input_names_; }
                std::vector<std::string> OutputNames() const override { return output_names_; }

            private:
                Ort::Env env_;
                Ort::Session* session_;
                Ort::SessionOptions session_options_;
                std::vector<std::string> input_names_;
                std::vector<std::string> output_names_;
        };
    }
}
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

// TensorRT is only available on Linux/Windows, not macOS
#ifdef __APPLE__
#warning "TensorRT is not supported on macOS. This file will be empty on macOS builds."
#else

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "engines/engine.h"
#include "engines/engine_config.h"
#include "runtime/data/tensor.h"

namespace ptk::perception
{

    class TrtEngine : public Engine
    {
    public:
        explicit TrtEngine(const EngineConfig &config);
        ~TrtEngine() override;

        bool Load(const std::string &engine_path) override;

        bool Infer(const std::vector<data::TensorView> &inputs,
                   std::vector<data::TensorView> &outputs) override;

        std::vector<std::string> InputNames() const override { return input_names_; }
        std::vector<std::string> OutputNames() const override { return output_names_; }

    private:
        struct TensorBinding
        {
            std::string name;
            int index = -1;
            nvinfer1::DataType trt_dtype;
            std::vector<int64_t> dims;
            size_t bytes = 0;
            void *device_ptr = nullptr;
        };

        EngineConfig config_;

        // TensorRT objects
        nvinfer1::IRuntime *runtime_ = nullptr;
        nvinfer1::ICudaEngine *engine_ = nullptr;
        nvinfer1::IExecutionContext *context_ = nullptr;

        cudaStream_t stream_ = nullptr;

        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;

        std::unordered_map<std::string, TensorBinding> bindings_;

        // helpers
        bool AllocateBindings();
        bool SetBindingDimensions(const std::vector<data::TensorView> &inputs);
        bool CopyInputsToDevice(const std::vector<data::TensorView> &inputs);
        bool CopyOutputsToHost(std::vector<data::TensorView> &outputs);

        size_t ElementSize(nvinfer1::DataType t) const;
    };

} // namespace ptk::perception

#endif // !__APPLE__
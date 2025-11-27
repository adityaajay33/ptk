#pragma once

#include <string>

namespace ptk::perception {

    enum class EngineBackend {
        OnnxRuntime,
        TensorRTNative
    };

    enum class OnnxRuntimeExecutionProvider {
        Cpu,
        Cuda,
        TensorRTEP
    };

    enum class TensorRTPrecisionMode {
        FP32,
        FP16,
        INT8
    };

    struct EngineConfig {
        EngineBackend backend = EngineBackend::OnnxRuntime;

        OnnxRuntimeExecutionProvider onnx_execution_provider = OnnxRuntimeExecutionProvider::Cpu;

        TensorRTPrecisionMode tensorrt_precision_mode = TensorRTPrecisionMode::FP32;

        int device_id = 0;
        bool enable_dynamic_shapes = false;
        size_t trt_workspace_size_mb = 1024;
        bool verbose = false;
        std::string trt_engine_path;
    };

}
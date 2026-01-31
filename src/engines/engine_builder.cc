#include "engines/engine_builder.h"
#include "engines/onnx_engine.h"
#include "engines/trt_engine.h"

namespace ptk::perception
{

    std::unique_ptr<Engine> CreateEngine(const EngineConfig &config)
    {
        if (config.backend == EngineBackend::OnnxRuntime)
        {
            // Create OnnxEngine with the provided config
            return std::make_unique<OnnxEngine>(config);
        }

        if (config.backend == EngineBackend::TensorRTNative)
        {
            #ifdef PTK_ENABLE_CUDA
            return std::make_unique<TrtEngine>(config);
            #else
            return nullptr; // TensorRT requires CUDA
            #endif
        }

        return nullptr;
    }

}
#include "engines/onnx_engine.h"
#include "engines/onnx_utils.h"
#include "runtime/core/logger.h"

#include <iostream>
#include <cstring>
#include <cstdlib>

namespace ptk::perception
{

    OnnxEngine::OnnxEngine(const EngineConfig &config)
        : Engine(),
          env_(ORT_LOGGING_LEVEL_WARNING, "ptk-onnx"),
          config_(config),
          active_provider_(OnnxRuntimeExecutionProvider::Cpu) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        core::LoggerManager::Info("OnnxEngine", "Initialized with " + std::string(config.verbose ? "verbose" : "normal") + " logging");
    }

    OnnxEngine::~OnnxEngine() = default;

    core::EngineStatus OnnxEngine::Load(const std::string &model_path)
    {
        core::LoggerManager::Info("OnnxEngine", "Loading model from: " + model_path);
        
        // Initialize execution providers based on config
        auto ep_status = InitializeExecutionProviders();
        if (!ep_status.ok()) {
            core::LoggerManager::Error("OnnxEngine", ep_status.message());
            return ep_status;
        }

        try
        {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        }
        catch (const Ort::Exception &e)
        {
            auto status = core::EngineStatus::ModelLoadFailed(e.what());
            core::LoggerManager::Error("OnnxEngine", status.message());
            return status;
        }

        core::LoggerManager::Info("OnnxEngine", "Model loaded successfully");

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);

        for (size_t i = 0; i < num_inputs; i++)
        {
            char *name = session_->GetInputNameAllocated(i, allocator).get();
            if (name)
            {
                input_names_.push_back(std::string(name));
            }
        }

        size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);

        for (size_t i = 0; i < num_outputs; i++)
        {
            char *name = session_->GetOutputNameAllocated(i, allocator).get();
            if (name)
            {
                output_names_.push_back(std::string(name));
            }
        }

        core::LoggerManager::Info("OnnxEngine", "Loaded " + std::to_string(num_inputs) + 
                                   " inputs and " + std::to_string(num_outputs) + " outputs");
        return core::EngineStatus::Ok();
    }

    Ort::Value OnnxEngine::CreateOrtTensorFromPtk(const data::TensorView &tv)
    {
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

        std::vector<int64_t> shape = tv.shape().dims();
        size_t total_bytes = tv.buffer().size_bytes();
        void *data_ptr = const_cast<void *>(tv.buffer().data());

        return Ort::Value::CreateTensor(mem_info, data_ptr, total_bytes, shape.data(), shape.size(), ptk::onnx::OnnxTypeFromPtkType(tv.dtype()));
    }

    core::EngineStatus OnnxEngine::Infer(const std::vector<data::TensorView> &inputs,
                           std::vector<data::TensorView> &outputs)
    {
        if (!session_)
        {
            auto status = core::EngineStatus(core::EngineErrorCode::kSessionNotLoaded,
                                            "Session not loaded");
            core::LoggerManager::Error("OnnxEngine", status.message());
            return status;
        }

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(inputs.size());

        for (const auto &tv : inputs)
        {
            ort_inputs.emplace_back(CreateOrtTensorFromPtk(tv));
        }

        std::vector<const char *> input_names_c;
        for (auto &s : input_names_)
            input_names_c.push_back(s.c_str());

        std::vector<const char *> output_names_c;
        for (auto &s : output_names_)
            output_names_c.push_back(s.c_str());

        std::vector<Ort::Value> ort_outputs;
        try
        {
            ort_outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(), output_names_c.data(), output_names_c.size());
        }
        catch (const Ort::Exception &e)
        {
            auto status = core::EngineStatus::InferenceFailed(e.what());
            core::LoggerManager::Error("OnnxEngine", status.message());
            return status;
        }

        outputs.clear();
        outputs.reserve(ort_outputs.size());

        for (size_t i = 0; i < ort_outputs.size(); i++)
        {
            auto &v = ort_outputs[i];

            Ort::TensorTypeAndShapeInfo info = v.GetTensorTypeAndShapeInfo();
            auto shape_vec = info.GetShape();
            ONNXTensorElementDataType elem_type = info.GetElementType();

            size_t elem_count = info.GetElementCount();
            size_t bytes_per_elem = 0;

            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                bytes_per_elem = 4;
            }
            else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
            {
                bytes_per_elem = 8;
            }
            else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
            {
                bytes_per_elem = 4;
            }
            else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                bytes_per_elem = 8;
            }
            else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
            {
                bytes_per_elem = 1;
            }
            else
            {
                auto status = core::EngineStatus(core::EngineErrorCode::kTypeMismatch,
                                                "Unsupported output element type");
                core::LoggerManager::Error("OnnxEngine", status.message());
                return status;
            }

            size_t total_bytes = elem_count * bytes_per_elem;

            void *buf = std::malloc(total_bytes);
            if (!buf)
            {
                auto status = core::EngineStatus(core::EngineErrorCode::kMemoryAllocationFailed,
                                                "Failed to allocate output buffer");
                core::LoggerManager::Error("OnnxEngine", status.message());
                return status;
            }

            const void *ort_data = v.GetTensorRawData();
            std::memcpy(buf, ort_data, total_bytes);

            data::BufferView bv(buf, total_bytes, core::DeviceType::kCpu);
            data::TensorShape ts(shape_vec);
            core::DataType dtype = ptk::onnx::PtkTypeFromOnnx(elem_type);

            data::TensorView tv(bv, dtype, ts);
            outputs.push_back(tv);
        }

        core::LoggerManager::Debug("OnnxEngine", "Inference completed successfully");
        return core::EngineStatus::Ok();
    }

    core::EngineStatus OnnxEngine::InitializeExecutionProviders() {
        core::LoggerManager::Info("OnnxEngine", "Initializing execution providers...");

        switch (config_.onnx_execution_provider) {
            case OnnxRuntimeExecutionProvider::Cuda:
                if (TryInitializeCudaProvider()) {
                    active_provider_ = OnnxRuntimeExecutionProvider::Cuda;
                    core::LoggerManager::Info("OnnxEngine", "CUDA provider initialized successfully");
                    return core::EngineStatus::Ok();
                }
                core::LoggerManager::Warn("OnnxEngine", "CUDA provider unavailable, falling back to CPU");
                return core::EngineStatus::Ok();  // CPU fallback

            case OnnxRuntimeExecutionProvider::TensorRTEP:
                if (TryInitializeTensorRtProvider()) {
                    active_provider_ = OnnxRuntimeExecutionProvider::TensorRTEP;
                    core::LoggerManager::Info("OnnxEngine", "TensorRT EP initialized successfully");
                    return core::EngineStatus::Ok();
                }
                core::LoggerManager::Warn("OnnxEngine", "TensorRT EP unavailable, trying CUDA");
                if (TryInitializeCudaProvider()) {
                    active_provider_ = OnnxRuntimeExecutionProvider::Cuda;
                    core::LoggerManager::Info("OnnxEngine", "Fell back to CUDA provider");
                    return core::EngineStatus::Ok();
                }
                core::LoggerManager::Warn("OnnxEngine", "Falling back to CPU");
                return core::EngineStatus::Ok();  // CPU fallback

            case OnnxRuntimeExecutionProvider::Cpu:
            default:
                return core::EngineStatus::Ok();  // CPU always available
        }
    }

    bool OnnxEngine::TryInitializeCudaProvider() {
#ifdef __APPLE__
        if (config_.verbose) {
            std::cout << "OnnxEngine: CUDA not available on macOS\n";
        }
        return false;
#else
        try {
            // Configure CUDA provider options
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config_.device_id;

            // Append CUDA provider with fallback
            session_options_.AppendExecutionProvider_CUDA(cuda_options);

            if (config_.verbose) {
                std::cout << "OnnxEngine: CUDA provider configured (device " << config_.device_id << ")\n";
            }
            return true;
        } catch (const std::exception &e) {
            if (config_.verbose) {
                std::cout << "OnnxEngine: Failed to initialize CUDA provider: " << e.what() << "\n";
            }
            return false;
        }
#endif
    }

    bool OnnxEngine::TryInitializeTensorRtProvider() {
#ifdef __APPLE__
        if (config_.verbose) {
            std::cout << "OnnxEngine: TensorRT EP not available on macOS\n";
        }
        return false;
#else
        try {
            // Configure TensorRT provider options
            OrtTensorRTProviderOptions trt_options;
            trt_options.device_id = config_.device_id;

            // Set precision mode
            switch (config_.tensorrt_precision_mode) {
                case TensorRTPrecisionMode::FP16:
                    trt_options.trt_fp16_enable = 1;
                    break;
                case TensorRTPrecisionMode::INT8:
                    trt_options.trt_int8_enable = 1;
                    break;
                case TensorRTPrecisionMode::FP32:
                default:
                    break;
            }

            // Set workspace size (in bytes)
            trt_options.trt_max_workspace_size = static_cast<size_t>(config_.trt_workspace_size_mb) * 1024 * 1024;

            // Append TensorRT provider with fallback to CUDA
            session_options_.AppendExecutionProvider_TensorRT(trt_options);

            if (config_.verbose) {
                std::cout << "OnnxEngine: TensorRT EP configured (device " << config_.device_id << ", "
                          << "workspace " << config_.trt_workspace_size_mb << "MB)\n";
            }
            return true;
        } catch (const std::exception &e) {
            if (config_.verbose) {
                std::cout << "OnnxEngine: Failed to initialize TensorRT provider: " << e.what() << "\n";
            }
            return false;
        }
#endif
    }

    bool OnnxEngine::TryInitializeCpuProvider() {
        try {
            // CPU provider is always available, no explicit initialization needed
            // It's the default fallback
            active_provider_ = OnnxRuntimeExecutionProvider::Cpu;

            if (config_.verbose) {
                std::cout << "OnnxEngine: CPU provider initialized\n";
            }
            return true;
        } catch (const std::exception &e) {
            std::cerr << "OnnxEngine: Failed to initialize CPU provider: " << e.what() << "\n";
            return false;
        }
    }

    core::EngineStatus OnnxEngine::InitializeProvider(OnnxRuntimeExecutionProvider provider) {
        config_.onnx_execution_provider = provider;
        return InitializeExecutionProviders();
    }

} // namespace ptk::perception
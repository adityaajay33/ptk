#include "engines/trt_engine.h"

#ifndef __APPLE__

#include "engines/tensorrt_utils.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstring>

namespace ptk::perception
{
    TrtEngine::TrtEngine(const EngineConfig &config)
        : Engine(),
          config_(config),
          runtime_(nullptr),
          engine_(nullptr),
          context_(nullptr),
          stream_(nullptr),
          input_names_(),
          output_names_(),
          bindings_() {}

    TrtEngine::~TrtEngine()
    {
        if (context_)
        {
            context_->destroy();
            context_ = nullptr;
        }
        if (engine_)
        {
            engine_->destroy();
            engine_ = nullptr;
        }
        if (runtime_)
        {
            runtime_->destroy();
            runtime_ = nullptr;
        }
        if (stream_)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

        for (auto &binding : bindings_)
        {
            if (binding.second.device_ptr != nullptr)
            {
                cudaFree(binding.second.device_ptr);
                binding.second.device_ptr = nullptr;
            }
        }

        bindings_.clear();
    }

    bool TrtEngine::Load(const std::string &engine_path)
    {
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file)
        {
            std::cerr << "TrtEngine::Load - Failed to open engine file: " << engine_path << "\n";
            return false;
        }

        engine_file.seekg(0, std::ios::end);
        const size_t model_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);

        std::vector<char> model_data(model_size);
        engine_file.read(model_data.data(), model_size);
        engine_file.close();

        runtime_ = nvinfer1::createInferRuntime(gLogger_);
        if (!runtime_)
        {
            std::cerr << "TrtEngine::Load - Failed to create TensorRT runtime\n";
            return false;
        }

        engine_ = runtime_->deserializeCudaEngine(model_data.data(), model_size);
        if (!engine_)
        {
            std::cerr << "TrtEngine::Load - Failed to deserialize CUDA engine\n";
            return false;
        }

        context_ = engine_->createExecutionContext();
        if (!context_)
        {
            std::cerr << "TrtEngine::Load - Failed to create execution context\n";
            return false;
        }

        cudaError_t cuda_status = cudaStreamCreate(&stream_);
        if (cuda_status != cudaSuccess)
        {
            std::cerr << "TrtEngine: Failed to create CUDA stream: "
                      << cudaGetErrorString(cuda_status) << "\n";
            return false;
        }

        const int32_t num_bindings = engine_->getNbBindings();

        for (int32_t i = 0; i < num_bindings; ++i)
        {
            const char *binding_name = engine_->getBindingName(i);
            const bool is_input = engine_->bindingIsInput(i);

            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            nvinfer1::DataType trt_dtype = engine_->getBindingDataType(i);

            TensorBinding binding;
            binding.name = binding_name;
            binding.index = i;
            binding.trt_dtype = trt_dtype;

            for (int32_t j = 0; j < dims.nbDims; ++j)
            {
                binding.dims.push_back(dims.d[j]);
            }

            binding.bytes = 1;
            for (const auto &dim : binding.dims)
            {
                binding.bytes *= dim;
            }
            binding.bytes *= static_cast<size_t>(ptk::trt::TrtElementSize(trt_dtype));

            bindings_[binding.name] = binding;

            if (is_input)
            {
                input_names_.push_back(binding.name);
            }
            else
            {
                output_names_.push_back(binding.name);
            }
        }

        if (config_.verbose)
        {
            std::cout << "TrtEngine: Loaded engine with " << input_names_.size()
                      << " inputs and " << output_names_.size() << " outputs\n";
        }

        if (!AllocateBindings())
        {
            std::cerr << "TrtEngine: Failed to allocate device memory\n";
            return false;
        }

        return true;
    }

    bool TrtEngine::Infer(const std::vector<data::TensorView> &inputs, std::vector<data::TensorView> &outputs)
    {
        if (!context_ || !engine_)
        {
            std::cerr << "TrtEngine::Infer - Engine not loaded properly\n";
            return false;
        }

        if (config_.enable_dynamic_shapes)
        {
            if (!SetBindingDimensions(inputs))
            {
                std::cerr << "TrtEngine::Infer - Failed to set binding dimensions\n";
                return false;
            }
        }

        if (!CopyInputsToDevice(inputs))
        {
            std::cerr << "TrtEngine::Infer - Failed to copy inputs to device\n";
            return false;
        }

        std::vector<void *> device_bindings(engine_->getNbBindings(), nullptr);

        for (const auto &binding : bindings_)
        {
            device_bindings[binding.second.index] = binding.second.device_ptr;
        }

        const bool success = context_->enqueueV3(stream_);
        if (!success)
        {
            std::cerr << "TrtEngine: Failed to enqueue inference\n";
            return false;
        }

        cudaError_t cuda_status = cudaStreamSynchronize(stream_);
        if (cuda_status != cudaSuccess)
        {
            std::cerr << "TrtEngine: CUDA stream synchronization failed: "
                      << cudaGetErrorString(cuda_status) << "\n";
            return false;
        }

        if (!CopyOutputsToHost(outputs))
        {
            std::cerr << "TrtEngine: Failed to copy outputs to host\n";
            return false;
        }

        return true;
    }

    bool TrtEngine::AllocateBindings()
    {
        for (auto &binding : bindings_)
        {
            if (binding.second.bytes == 0)
            {
                std::cerr << "TrtEngine: Binding " << binding.first
                          << " has zero size\n";
                return false;
            }

            cudaError_t cuda_status =
                cudaMalloc(&binding.second.device_ptr, binding.second.bytes);
            if (cuda_status != cudaSuccess)
            {
                std::cerr << "TrtEngine: cudaMalloc failed for binding "
                          << binding.first << ": " << cudaGetErrorString(cuda_status)
                          << "\n";
                return false;
            }

            if (config_.verbose)
            {
                std::cout << "TrtEngine: Allocated " << binding.second.bytes
                          << " bytes for binding " << binding.first << "\n";
            }
        }

        return true;
    }

    bool TrtEngine::SetBindingDimensions(const std::vector<data::TensorView> &inputs)
    {
        if (inputs.size() != input_names_.size())
        {
            std::cerr << "TrtEngine: Expected " << input_names_.size()
                      << " inputs, got " << inputs.size() << "\n";
            return false;
        }

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto &input = inputs[i];
            const auto &input_name = input_names_[i];

            auto it = bindings_.find(input_name);
            if (it == bindings_.end())
            {
                std::cerr << "TrtEngine: Unknown input binding: " << input_name << "\n";
                return false;
            }

            const auto &shape_dims = input.shape().dims();
            nvinfer1::Dims trt_dims;
            trt_dims.nbDims = shape_dims.size();

            for (size_t j = 0; j < shape_dims.size(); ++j)
            {
                trt_dims.d[j] = shape_dims[j];
            }

            const bool success = context_->setInputShape(input_name.c_str(), trt_dims);
            if (!success)
            {
                std::cerr << "TrtEngine: Failed to set input shape for " << input_name
                          << "\n";
                return false;
            }

            it->second.dims.clear();
            for (int32_t j = 0; j < trt_dims.nbDims; ++j)
            {
                it->second.dims.push_back(trt_dims.d[j]);
            }

            size_t total_elements = 1;
            for (const auto &dim : it->second.dims)
            {
                total_elements *= dim;
            }
            it->second.bytes =
                total_elements * ptk::trt::TrtElementSize(it->second.trt_dtype);
        }

        return true;
    }

    bool TrtEngine::CopyInputsToDevice(
        const std::vector<data::TensorView> &inputs)
    {
        if (inputs.size() != input_names_.size())
        {
            std::cerr << "TrtEngine: Expected " << input_names_.size()
                      << " inputs, got " << inputs.size() << "\n";
            return false;
        }

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const auto &input = inputs[i];
            const auto &input_name = input_names_[i];

            auto it = bindings_.find(input_name);
            if (it == bindings_.end())
            {
                std::cerr << "TrtEngine: Unknown input binding: " << input_name << "\n";
                return false;
            }

            const size_t input_bytes = input.buffer().size_bytes();
            if (input_bytes != it->second.bytes)
            {
                std::cerr << "TrtEngine: Input " << input_name << " size mismatch: "
                          << "expected " << it->second.bytes << " bytes, got "
                          << input_bytes << " bytes\n";
                return false;
            }

            cudaError_t cuda_status = cudaMemcpyAsync(
                it->second.device_ptr,
                const_cast<void *>(input.buffer().data()),
                input_bytes,
                cudaMemcpyHostToDevice,
                stream_);

            if (cuda_status != cudaSuccess)
            {
                std::cerr << "TrtEngine: H2D memcpy failed for " << input_name << ": "
                          << cudaGetErrorString(cuda_status) << "\n";
                return false;
            }
        }

        return true;
    }

    bool TrtEngine::CopyOutputsToHost(
        std::vector<data::TensorView> &outputs)
    {
        outputs.clear();
        outputs.reserve(output_names_.size());

        for (const auto &output_name : output_names_)
        {
            auto it = bindings_.find(output_name);
            if (it == bindings_.end())
            {
                std::cerr << "TrtEngine: Unknown output binding: " << output_name
                          << "\n";
                return false;
            }

            const auto &binding = it->second;

            // Allocate host memory for output
            void *host_buffer = std::malloc(binding.bytes);
            if (!host_buffer)
            {
                std::cerr << "TrtEngine: Failed to allocate host memory for output "
                          << output_name << "\n";

                return false;
            }

            // Copy from device to host
            cudaError_t cuda_status = cudaMemcpyAsync(
                host_buffer,
                binding.device_ptr,
                binding.bytes,
                cudaMemcpyDeviceToHost,
                stream_);

            if (cuda_status != cudaSuccess)
            {
                std::cerr << "TrtEngine: D2H memcpy failed for " << output_name << ": "
                          << cudaGetErrorString(cuda_status) << "\n";
                std::free(host_buffer);
                return false;
            }

            // Create TensorView for output
            data::BufferView buffer_view(
                host_buffer,
                binding.bytes,
                core::DeviceType::kCpu);

            core::DataType ptk_dtype = ptk::onnx::PtkTypeFromOnnx(
                static_cast<ONNXTensorElementDataType>(binding.trt_dtype));

            data::TensorShape shape(binding.dims);
            data::TensorView output_view(buffer_view, ptk_dtype, shape);

            outputs.push_back(output_view);
        }
        return true;
    }

    size_t TrtEngine::ElementSize(nvinfer1::DataType dtype) const
    {
        return ptk::trt::TrtElementSize(dtype);
    }

    // Static logger initialization
    TrtLogger TrtEngine::gLogger_;

} // namespace ptk::perception

#endif // !__APPLE__
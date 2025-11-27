#include "engines/onnx_engine.h"
#include "engines/onnx_utils.h"

#include <iostream>
#include <cstring>
#include <cstdlib>

namespace ptk::perception
{

    OnnxEngine::OnnxEngine(const EngineConfig &config) : Engine(), env_(ORT_LOGGING_LEVEL_WARNING, "ptk-onnx"), config_(config)
    {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    OnnxEngine::~OnnxEngine() = default;

    bool OnnxEngine::Load(const std::string &model_path)
    {
        try
        {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "ONNX Load Error: " << e.what() << "\n";
            return false;
        }

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

        return true;
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

    bool OnnxEngine::Infer(const std::vector<data::TensorView> &inputs,
                           std::vector<data::TensorView> &outputs)
    {
        if (!session_)
        {
            std::cerr << "OnnxEngine: Session not loaded\n";
            return false;
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
            std::cerr << "ONNX Infer Error: " << e.what() << "\n";
            return false;
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
                std::cerr << "OnnxEngine: Unsupported output element type\n";
                return false;
            }

            size_t total_bytes = elem_count * bytes_per_elem;

            void *buf = std::malloc(total_bytes);
            if (!buf)
            {
                std::cerr << "OnnxEngine: Failed to allocate output buffer\n";
                return false;
            }

            const void *ort_data = v.GetTensorRawData();
            std::memcpy(buf, ort_data, total_bytes);

            data::BufferView bv(buf, total_bytes, core::DeviceType::kCpu);
            data::TensorShape ts(shape_vec);
            core::DataType dtype = ptk::onnx::PtkTypeFromOnnx(elem_type);

            data::TensorView tv(bv, dtype, ts);
            outputs.push_back(tv);
        }

        return true;
    }

} // namespace ptk::perception
#include "engines/engine_validation.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef __APPLE__
#include <sys/resource.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace ptk::validation
{

    // EngineValidator Implementation

    EngineValidator::EngineValidator(ptk::perception::Engine *engine)
        : engine_(engine) {}

    core::Status EngineValidator::ValidateModelNames(
        const std::vector<std::string> &expected_input_names,
        const std::vector<std::string> &expected_output_names) const
    {
        if (!engine_)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "EngineValidator: Engine is null");
        }

        auto actual_inputs = engine_->InputNames();
        auto actual_outputs = engine_->OutputNames();

        // Check input names
        if (actual_inputs.size() != expected_input_names.size())
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "EngineValidator: Input count mismatch");
        }

        for (size_t i = 0; i < expected_input_names.size(); ++i)
        {
            if (actual_inputs[i] != expected_input_names[i])
            {
                return core::Status(
                    core::StatusCode::kInvalidArgument,
                    "EngineValidator: Input name mismatch at index " +
                        std::to_string(i));
            }
        }

        // Check output names
        if (actual_outputs.size() != expected_output_names.size())
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "EngineValidator: Output count mismatch");
        }

        for (size_t i = 0; i < expected_output_names.size(); ++i)
        {
            if (actual_outputs[i] != expected_output_names[i])
            {
                return core::Status(
                    core::StatusCode::kInvalidArgument,
                    "EngineValidator: Output name mismatch at index " +
                        std::to_string(i));
            }
        }

        return core::Status::Ok();
    }

    core::Status EngineValidator::ValidateModelShape(
        const std::vector<ShapeExpectation> &expectations) const
    {
        if (!engine_)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "EngineValidator: Engine is null");
        }

        auto input_names = engine_->InputNames();
        auto output_names = engine_->OutputNames();

        for (const auto &expectation : expectations)
        {
            // Find the tensor in inputs or outputs
            bool found = false;

            if (expectation.is_input)
            {
                for (const auto &name : input_names)
                {
                    if (name == expectation.name)
                    {
                        found = true;
                        break;
                    }
                }
            }
            else
            {
                for (const auto &name : output_names)
                {
                    if (name == expectation.name)
                    {
                        found = true;
                        break;
                    }
                }
            }

            if (!found)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                                    "EngineValidator: Tensor not found: " +
                                        expectation.name);
            }
        }

        return core::Status::Ok();
    }

    void EngineValidator::LogEngineConfig(
        const ptk::perception::EngineConfig &config) const
    {
        ConfigLogger::Log(config, config.verbose);
    }

    BenchmarkResult EngineValidator::BenchmarkInference(
        const std::vector<data::TensorView> &sample_inputs,
        size_t num_iterations) const
    {
        if (!engine_)
        {
            BenchmarkResult empty;
            return empty;
        }

        return BenchmarkUtility::MeasureLatency(engine_, sample_inputs,
                                                num_iterations);
    }

    std::string EngineValidator::GetEngineSummary() const
    {
        if (!engine_)
        {
            return "Engine: NULL";
        }

        std::ostringstream oss;
        oss << "Engine Summary:\n";
        oss << "  Inputs: " << engine_->InputNames().size() << "\n";
        for (const auto &name : engine_->InputNames())
        {
            oss << "    - " << name << "\n";
        }
        oss << "  Outputs: " << engine_->OutputNames().size() << "\n";
        for (const auto &name : engine_->OutputNames())
        {
            oss << "    - " << name << "\n";
        }

        return oss.str();
    }

    // ============================================================================
    // ConfigLogger Implementation
    // ============================================================================

    std::string ConfigLogger::FormatEngineConfig(
        const ptk::perception::EngineConfig &config)
    {
        std::ostringstream oss;

        oss << "Engine Configuration:\n";
        oss << "  Backend: "
            << (config.backend == ptk::perception::EngineBackend::OnnxRuntime
                    ? "ONNX Runtime"
                    : "TensorRT Native")
            << "\n";

        oss << "  Execution Provider: " << FormatExecutionProvider(config.onnx_execution_provider)
            << "\n";

        oss << "  Device ID: " << config.device_id << "\n";
        oss << "  Dynamic Shapes: " << (config.enable_dynamic_shapes ? "enabled" : "disabled")
            << "\n";

        if (config.backend == ptk::perception::EngineBackend::OnnxRuntime)
        {
            oss << "  Precision Mode: "
                << FormatPrecisionMode(config.tensorrt_precision_mode) << "\n";
            oss << "  TensorRT Workspace: " << config.trt_workspace_size_mb << " MB\n";
        }

        oss << "  Verbose: " << (config.verbose ? "enabled" : "disabled") << "\n";

        return oss.str();
    }

    std::string ConfigLogger::FormatExecutionProvider(
        ptk::perception::OnnxRuntimeExecutionProvider provider)
    {
        switch (provider)
        {
        case ptk::perception::OnnxRuntimeExecutionProvider::Cpu:
            return "CPU";
        case ptk::perception::OnnxRuntimeExecutionProvider::Cuda:
            return "CUDA";
        case ptk::perception::OnnxRuntimeExecutionProvider::TensorRTEP:
            return "TensorRT EP";
        default:
            return "Unknown";
        }
    }

    std::string ConfigLogger::FormatPrecisionMode(
        ptk::perception::TensorRTPrecisionMode mode)
    {
        switch (mode)
        {
        case ptk::perception::TensorRTPrecisionMode::FP32:
            return "FP32 (Full Precision)";
        case ptk::perception::TensorRTPrecisionMode::FP16:
            return "FP16 (Half Precision)";
        case ptk::perception::TensorRTPrecisionMode::INT8:
            return "INT8 (Quantized)";
        default:
            return "Unknown";
        }
    }

    std::string ConfigLogger::FormatTensorShape(const data::TensorShape &shape)
    {
        std::ostringstream oss;
        oss << "[";
        const auto &dims = shape.dims();
        for (size_t i = 0; i < dims.size(); ++i)
        {
            oss << dims[i];
            if (i < dims.size() - 1)
                oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

    void ConfigLogger::Log(const ptk::perception::EngineConfig &config,
                           bool verbose)
    {
        if (!verbose)
            return;

        std::cout << FormatEngineConfig(config) << std::flush;
    }

    // ============================================================================
    // BenchmarkUtility Implementation
    // ============================================================================

    BenchmarkResult BenchmarkUtility::MeasureLatency(
        ptk::perception::Engine *engine,
        const std::vector<data::TensorView> &inputs,
        size_t num_iterations,
        size_t warmup_iterations)
    {
        BenchmarkResult result;
        result.num_iterations = num_iterations;

        if (!engine || inputs.empty())
        {
            return result;
        }

        std::vector<data::TensorView> outputs;

        // Warmup iterations (not counted)
        for (size_t w = 0; w < warmup_iterations; ++w)
        {
            engine->Infer(inputs, outputs);
        }

        std::vector<double> latencies_ms;
        latencies_ms.reserve(num_iterations);

        // Timed iterations
        for (size_t i = 0; i < num_iterations; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            engine->Infer(inputs, outputs);
            auto end = std::chrono::high_resolution_clock::now();

            double latency_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
            latencies_ms.push_back(latency_ms);
        }

        // Calculate statistics
        if (!latencies_ms.empty())
        {
            result.min_latency_ms =
                *std::min_element(latencies_ms.begin(), latencies_ms.end());
            result.max_latency_ms =
                *std::max_element(latencies_ms.begin(), latencies_ms.end());

            double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
            result.mean_latency_ms = sum / latencies_ms.size();

            // Standard deviation
            double variance = 0.0;
            for (double lat : latencies_ms)
            {
                variance += (lat - result.mean_latency_ms) * (lat - result.mean_latency_ms);
            }
            variance /= latencies_ms.size();
            result.std_dev_ms = std::sqrt(variance);

            // Throughput (FPS)
            result.throughput_fps = 1000.0 / result.mean_latency_ms;
        }

        return result;
    }

    double BenchmarkUtility::MeasureThroughput(
        ptk::perception::Engine *engine,
        const std::vector<data::TensorView> &inputs,
        size_t duration_seconds)
    {
        if (!engine || inputs.empty())
        {
            return 0.0;
        }

        std::vector<data::TensorView> outputs;
        size_t frame_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time =
            start_time + std::chrono::seconds(static_cast<int>(duration_seconds));

        while (std::chrono::high_resolution_clock::now() < end_time)
        {
            engine->Infer(inputs, outputs);
            frame_count++;
        }

        double elapsed_seconds = std::chrono::duration<double>(
                                     std::chrono::high_resolution_clock::now() -
                                     start_time)
                                     .count();
        return frame_count / elapsed_seconds;
    }

    MemoryUsage BenchmarkUtility::MeasureMemoryUsage(
        ptk::perception::Engine *engine)
    {
        MemoryUsage usage;

        // CPU memory measurement (platform-dependent)
#ifdef __linux__
        // Linux: read from /proc/self/status
        FILE *fp = fopen("/proc/self/status", "r");
        if (fp)
        {
            char line[256];
            while (fgets(line, sizeof(line), fp))
            {
                if (sscanf(line, "VmRSS: %lf kB", &usage.cpu_mb) == 1)
                {
                    usage.cpu_mb /= 1024.0; // Convert KB to MB
                    break;
                }
            }
            fclose(fp);
        }
#elif defined(__APPLE__)
        // macOS: use getrusage
        struct rusage usage_info;
        getrusage(RUSAGE_SELF, &usage_info);
        usage.cpu_mb =
            static_cast<double>(usage_info.ru_maxrss) / 1024.0; // Convert bytes to MB
#elif defined(_WIN32)
        // Windows: use GetProcessMemoryInfo
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        {
            usage.cpu_mb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        }
#endif

        // GPU memory measurement (CUDA if available)
#ifndef __APPLE__
        // TODO: Implement cudaMemGetInfo() for GPU memory
        // cudaMemGetInfo(&free_mem, &total_mem);
        // usage.gpu_mb = (total_mem - free_mem) / (1024.0 * 1024.0);
#endif

        return usage;
    }

    std::string BenchmarkUtility::FormatBenchmarkResult(
        const BenchmarkResult &result)
    {
        std::ostringstream oss;

        oss << std::fixed << std::setprecision(3);
        oss << "Benchmark Results (" << result.num_iterations << " iterations):\n";
        oss << "  Mean Latency: " << result.mean_latency_ms << " ms\n";
        oss << "  Min Latency:  " << result.min_latency_ms << " ms\n";
        oss << "  Max Latency:  " << result.max_latency_ms << " ms\n";
        oss << "  Std Dev:      " << result.std_dev_ms << " ms\n";
        oss << "  Throughput:   " << result.throughput_fps << " FPS\n";

        return oss.str();
    }

} // namespace ptk::validation

#pragma once

#include "engine.h"
#include "engine_config.h"
#include "runtime/data/tensor.h"
#include "runtime/core/status.h"

#include <vector>
#include <string>
#include <chrono>

namespace ptk::validation
{

    // Validation Structures

    struct ShapeExpectation
    {
        std::string name;
        std::vector<int64_t> shape;
        bool is_input = true;
    };

    struct BenchmarkResult
    {
        double mean_latency_ms = 0.0;
        double min_latency_ms = 0.0;
        double max_latency_ms = 0.0;
        double std_dev_ms = 0.0;
        size_t num_iterations = 0;
        double throughput_fps = 0.0;
    };

    struct MemoryUsage
    {
        double cpu_mb = 0.0;
        double gpu_mb = 0.0;
    };

    // Engine Validation Utilities

    class EngineValidator
    {
    public:
        explicit EngineValidator(ptk::perception::Engine *engine);

        core::Status ValidateModelNames(
            const std::vector<std::string> &expected_input_names,
            const std::vector<std::string> &expected_output_names) const;

        core::Status ValidateModelShape(
            const std::vector<ShapeExpectation> &expectations) const;

        void LogEngineConfig(const ptk::perception::EngineConfig &config) const;

        BenchmarkResult BenchmarkInference(
            const std::vector<data::TensorView> &sample_inputs,
            size_t num_iterations = 100) const;

        std::string GetEngineSummary() const;

    private:
        ptk::perception::Engine *engine_;

        bool ValidateOnnxMetadata(const std::string &model_path) const;
    };

    class ConfigLogger
    {
    public:
        static std::string FormatEngineConfig(
            const ptk::perception::EngineConfig &config);

        static std::string FormatExecutionProvider(
            ptk::perception::OnnxRuntimeExecutionProvider provider);

        static std::string FormatPrecisionMode(
            ptk::perception::TensorRTPrecisionMode mode);

        static std::string FormatTensorShape(const data::TensorShape &shape);

        static void Log(const ptk::perception::EngineConfig &config,
                        bool verbose = true);
    };

    class BenchmarkUtility
    {
    public:
        static BenchmarkResult MeasureLatency(
            ptk::perception::Engine *engine,
            const std::vector<data::TensorView> &inputs,
            size_t num_iterations = 100,
            size_t warmup_iterations = 5);

        static double MeasureThroughput(
            ptk::perception::Engine *engine,
            const std::vector<data::TensorView> &inputs,
            size_t duration_seconds = 5);

        static MemoryUsage MeasureMemoryUsage(ptk::perception::Engine *engine);

        static std::string FormatBenchmarkResult(const BenchmarkResult &result);
    };

} // namespace ptk::validation

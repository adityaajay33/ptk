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

        // Validate that model input names match expected names
        core::Status ValidateModelNames(
            const std::vector<std::string> &expected_input_names,
            const std::vector<std::string> &expected_output_names) const;

        // Validate that input/output shapes match expected dimensions
        core::Status ValidateModelShape(
            const std::vector<ShapeExpectation> &expectations) const;

        // Log engine configuration for debugging
        void LogEngineConfig(const ptk::perception::EngineConfig &config) const;

        // Benchmark inference latency
        BenchmarkResult BenchmarkInference(
            const std::vector<data::TensorView> &sample_inputs,
            size_t num_iterations = 100) const;

        // Get summary of engine capabilities
        std::string GetEngineSummary() const;

    private:
        ptk::perception::Engine *engine_;

        // Helper: Parse ONNX model directly (optional, for advanced validation)
        bool ValidateOnnxMetadata(const std::string &model_path) const;
    };

    // Configuration Logging Utilities

    class ConfigLogger
    {
    public:
        // Pretty-print EngineConfig
        static std::string FormatEngineConfig(
            const ptk::perception::EngineConfig &config);

        // Pretty-print execution provider info
        static std::string FormatExecutionProvider(
            ptk::perception::OnnxRuntimeExecutionProvider provider);

        // Pretty-print precision mode
        static std::string FormatPrecisionMode(
            ptk::perception::TensorRTPrecisionMode mode);

        // Pretty-print tensor shape
        static std::string FormatTensorShape(const data::TensorShape &shape);

        // Log to stdout
        static void Log(const ptk::perception::EngineConfig &config,
                        bool verbose = true);
    };

    // Benchmark Utilities

    class BenchmarkUtility
    {
    public:
        // Measure inference latency with multiple iterations
        static BenchmarkResult MeasureLatency(
            ptk::perception::Engine *engine,
            const std::vector<data::TensorView> &inputs,
            size_t num_iterations = 100,
            size_t warmup_iterations = 5);

        // Measure inference throughput (FPS)
        static double MeasureThroughput(
            ptk::perception::Engine *engine,
            const std::vector<data::TensorView> &inputs,
            size_t duration_seconds = 5);

        // Measure memory usage (CPU and GPU if available)
        static MemoryUsage MeasureMemoryUsage(ptk::perception::Engine *engine);

        // Format benchmark result for display
        static std::string FormatBenchmarkResult(const BenchmarkResult &result);
    };

} // namespace ptk::validation

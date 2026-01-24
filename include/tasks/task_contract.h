#pragma once

#include "runtime/core/status.h"
#include "runtime/data/tensor.h"
#include "engines/engine.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace ptk::tasks
{
    struct TaskInput;
    struct TaskOutput;

    enum class CoordinateSystem {
        kImagePixels,
        kImageNormalized,
        kCenterNormalized
    };

    struct TaskSpec {
        std::string name;
        std::string description;

        struct InputSpec {
            std::string name;
            std::vector<int64_t> shape;
            core::DataType dtype;
            core::TensorLayout layout;
            bool allow_batch;
        };
        std::vector<InputSpec> input_specs;

        struct OutputSpec {
            std::string name;
            std::string semantic_meaning;  // e.g., "bounding boxes in XYXY format"
            CoordinateSystem coordinate_system;
            core::DataType dtype;
        };
        std::vector<OutputSpec> output_specs;

        struct Invariant {

            std::string description;
            std::function<core::Status(const TaskOutput&)> validator;
        };
        std::vector<Invariant> invariants;

        struct Metadata {

            float min_confidence = 0.0f;
            float max_confidence = 1.0f;
            int max_detections = -1;
            bool requires_nms = false;
            std::vector<std::string> classes;
        };
        Metadata metadata;
    };

    class TaskContract
    {
        public:
            virtual ~TaskContract() = default;

            virtual const TaskSpec& GetSpec() const = 0;

            virtual core::Status ValidateInput(const TaskInput& input) const = 0;

            virtual core::Status Execute(perception::Engine& engine, const TaskInput& input, TaskOutput& output) = 0;

            virtual core::Status ValidateOutput(const TaskOutput& output) const = 0;

            virtual core::Status ValidateModel(perception::Engine& engine) const = 0;

            virtual core::Status PostProcess(
                const std::vector<data::TensorView>& raw_outputs,
                const TaskInput& original_input,
                TaskOutput* result) = 0;

            virtual core::Status PreProcess(
                const TaskInput& input,
                std::vector<data::TensorView>* raw_inputs) = 0;

        protected:
            TaskSpec spec_;
    };
}
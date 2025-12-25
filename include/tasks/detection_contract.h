#pragma once

#include "tasks/task_contract.h"
#include "tasks/task_output.h"

namespace ptk::tasks
{

    class ObjectDetectionContract : public TaskContract
    {
    public:
        ObjectDetectionContract(
            const std::vector<std::string> &class_labels,
            CoordinateSystem output_coords = CoordinateSystem::kImagePixels);

        ~ObjectDetectionContract() override = default;

        const TaskSpec &GetSpec() const override { return spec_; }

        core::Status ValidateInput(const TaskInput &input) const override;

        core::Status Execute(
            perception::Engine &engine,
            const TaskInput &input,
            TaskOutput &output) override;

        core::Status ValidateOutput(const TaskOutput &output) const override;

        core::Status ValidateModel(perception::Engine &engine) const override;

        core::Status PostProcess(
            const std::vector<data::TensorView> &raw_outputs,
            const TaskInput &original_input,
            TaskOutput *result) override;

        core::Status PreProcess(
            const TaskInput &input,
            std::vector<data::TensorView> *raw_inputs) override;

        core::Status ApplyNMS(
            std::vector<Detection> *detections,
            float iou_threshold) const;

        core::Status FilterByConfidence(
            std::vector<Detection> *detections,
            float threshold) const;

    private:
        std::vector<std::string> class_labels_;
        CoordinateSystem output_coords_;

        core::Status ValidateDetectionInvariants(const Detection &det) const;

        core::Status ParseYOLOOutput(
            const std::vector<data::TensorView> &outputs,
            TaskOutput *result);

        core::Status ParseFasterRCNNOutput(
            const std::vector<data::TensorView> &outputs,
            TaskOutput *result);
    };
}
#pragma once

#include "tasks/task_contract.h"
#include "tasks/task_output.h"

namespace ptk::tasks
{
    enum class SegmentationType {
        kInstance,
        kSemantic,
        kPanoptic
    };

    enum class MaskFormat {
        kClassIds,
        kProbabilities,
        kLogits,
        kBinaryMasks
    };
 
    class SegmentationContract : public TaskContract
    {
    public:
        SegmentationContract(
            const std::vector<std::string> &class_labels,
            SegmentationType seg_type = SegmentationType::kInstance,
            MaskFormat mask_format = MaskFormat::kClassIds);

        ~SegmentationContract() override = default;

        const TaskSpec &GetSpec() const override { return spec_; }

        core::Status ValidateInput(const TaskInput &input) const override;

        core::Status Execute(
            perception::Engine &engine,
            const TaskInput &input,
            TaskOutput &output) override;

        core::Status ValidateOutput(const TaskOutput &output) const override;

        core::Status ValidateModel(perception::Engine &engine) const override;

        core::Status ApplySoftmax(
            const data::TensorView &logits,
            data::TensorView *probabilities) const;

        core::Status ConvertLogitsToClassIds(
            const data::TensorView &logits,
            std::vector<uint8_t> *class_ids) const;

        core::Status ConvertProbabilitiesToClassIds(
            const data::TensorView &probabilities,
            std::vector<uint8_t> *class_ids) const;

        core::Status PostProcess(
            const std::vector<data::TensorView>& raw_outputs,
            const TaskInput& original_input,
            TaskOutput* result) override;

        core::Status PreProcess(
            const TaskInput& input,
            std::vector<data::TensorView>* raw_inputs) override;

        core::Status ResizeMaskToOriginalSize(
            const SegmentationMask &mask,
            int original_height,
            int original_width,
            SegmentationMask *resized_mask) const;

    private:
        std::vector<std::string> class_labels_;
        SegmentationType seg_type_;
        MaskFormat mask_format_;

        core::Status ParseClassIdMask(
            const std::vector<data::TensorView> &outputs,
            TaskOutput *result);

        core::Status ParseProbabilityMask(
            const std::vector<data::TensorView> &outputs,
            TaskOutput *result);

        core::Status ParseLogitMask(
            const std::vector<data::TensorView> &outputs,
            TaskOutput *result);

        core::Status ValidateMaskInvariants(const SegmentationMask &mask) const;
    };
}
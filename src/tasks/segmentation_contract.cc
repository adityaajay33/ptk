#include "tasks/segmentation_contract.h"
#include "runtime/core/logger.h"
#include "runtime/core/status.h"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace ptk::tasks
{

    SegmentationContract::SegmentationContract(const std::vector<std::string> &class_labels, SegmentationType seg_type, MaskFormat mask_format) : class_labels_(class_labels), seg_type_(seg_type), mask_format_(mask_format)
    {

        spec_.name = "Segmentation";
        spec_.description = "Segment objects in an image";

        TaskSpec::InputSpec input_spec;
        input_spec.name = "image";
        input_spec.shape = {-1, -1, 3}; //Only supports RGB images for now
        input_spec.dtype = core::DataType::kFloat32;
        input_spec.layout = core::TensorLayout::kHwc; //Only supports HWC layout for now 
        input_spec.allow_batch = true;
        spec_.input_specs.push_back(input_spec);

        TaskSpec::OutputSpec mask_spec;
        mask_spec.name = "segmentation mask";

        if (mask_format_ == MaskFormat::kClassIds)
        {
            mask_spec.semantic_meaning = "Class ID per pixel [H, W], values in [0, num_classes-1]";
            mask_spec.dtype = core::DataType::kUint8;
        }
        else if (mask_format_ == MaskFormat::kProbabilities)
        {
            mask_spec.semantic_meaning = "Probability maps [H, W, num_classes], values in [0.0, 1.0]";
            mask_spec.dtype = core::DataType::kFloat32;
        }
        else if (mask_format_ == MaskFormat::kLogits)
        {
            mask_spec.semantic_meaning = "Logit scores [H, W, num_classes], values in (-inf, +inf)";
            mask_spec.dtype = core::DataType::kFloat32;
        }

        mask_spec.coordinate_system = CoordinateSystem::kImagePixels;
        spec_.output_specs.push_back(mask_spec);

        spec_.metadata.classes = class_labels_;
        spec_.metadata.min_confidence = 0.0f;
        spec_.metadata.max_confidence = 1.0f;
        spec_.metadata.requires_nms = false; //We don't really need NMS

        TaskSpec::Invariant mask_shape_invariant;
        mask_shape_invariant.description =
            "Mask dimensions must match input image dimensions (or be downsampled by model factor)";
        mask_shape_invariant.validator = [](const TaskOutput &output) -> core::Status
        {
            if (!output.segmentation)
            {
                return core::Status(
                    core::StatusCode::kFailedPrecondition,
                    "Segmentation mask is missing");
            }
            return core::Status::Ok();
        };
        spec_.invariants.push_back(mask_shape_invariant);

        TaskSpec::Invariant class_id_invariant;
        class_id_invariant.description =
            "Class IDs must be in valid range [0, num_classes-1] or 255 (ignore)";
        class_id_invariant.validator = [this](const TaskOutput &output) -> core::Status
        {
            if (!output.segmentation || output.segmentation->mask.empty())
            {
                return core::Status::Ok();
            }

            int max_class_id = static_cast<int>(class_labels_.size() - 1);
            for (uint8_t class_id : output.segmentation->mask)
            {
                if (class_id > max_class_id && class_id != 255)
                {
                    return core::Status(
                        core::StatusCode::kFailedPrecondition,
                        "Invalid class ID in mask: " + std::to_string(class_id));
                }
            }
            return core::Status::Ok();
        };
        spec_.invariants.push_back(class_id_invariant);
    }

    core::Status SegmentationContract::ValidateInput(const TaskInput &input) const
    {
        if (input.frame.image.num_elements() == 0)
        {
            return core::Status(
                core::StatusCode::kInvalidArgument,
                "Input frame has no image data");
        }

        const auto &shape = input.frame.image.shape();
        if (shape.rank() != 3 && shape.rank() != 4)
        {
            return core::Status(
                core::StatusCode::kInvalidArgument,
                "Expected 3D (HWC) or 4D (NHWC) tensor");
        }

        if (shape.rank() == 3 && shape.dim(2) != 3)
        {
            return core::Status(
                core::StatusCode::kInvalidArgument,
                "Expected 3-channel image");
        }
        if (shape.rank() == 4 && shape.dim(3) != 3)
        {
            return core::Status(
                core::StatusCode::kInvalidArgument,
                "Expected 3-channel image in batch");
        }

        return core::Status::Ok();
    }

    core::Status SegmentationContract::Execute(
        perception::Engine &engine,
        const TaskInput &input,
        TaskOutput &output)
    {

        // Validate input
        core::Status status = ValidateInput(input);
        if (!status.ok())
        {
            return status;
        }

        // Preprocess
        std::vector<data::TensorView> raw_inputs;
        status = PreProcess(input, &raw_inputs);
        if (!status.ok())
        {
            return status;
        }

        // Execute inference
        std::vector<data::TensorView> raw_outputs;
        auto start = std::chrono::high_resolution_clock::now();

        auto engine_status = engine.Infer(raw_inputs, raw_outputs);
        if (!engine_status.ok())
        {
            return core::Status(
                core::StatusCode::kInternal,
                "Engine inference failed: " + engine_status.message());
        }

        auto end = std::chrono::high_resolution_clock::now();
        float duration_ms = std::chrono::duration<float, std::milli>(end - start).count();

        status = PostProcess(raw_outputs, input, &output);
        if (!status.ok())
        {
            return status;
        }

        output.task_type = spec_.name;
        output.inference_time_ms = duration_ms;
        output.timestamp_ns = input.frame.timestamp_ns;
        output.frame_index = input.frame.frame_index;
        output.success = true;

        status = ValidateOutput(output);
        if (!status.ok())
        {
            output.success = false;
            return status;
        }

        return core::Status::Ok();
    }

    core::Status SegmentationContract::PostProcess(
        const std::vector<data::TensorView> &raw_outputs,
        const TaskInput &original_input,
        TaskOutput *result)
    {

        if (raw_outputs.empty())
        {
            return core::Status(
                core::StatusCode::kInvalidArgument,
                "No output tensors from model");
        }

        // Parse based on mask format
        switch (mask_format_)
        {
        case MaskFormat::kClassIds:
            return ParseClassIdMask(raw_outputs, result);
        case MaskFormat::kProbabilities:
            return ParseProbabilityMask(raw_outputs, result);
        case MaskFormat::kLogits:
            return ParseLogitMask(raw_outputs, result);
        default:
            return core::Status(
                core::StatusCode::kInternal,
                "Unsupported mask format");
        }
    }

    core::Status SegmentationContract::ParseClassIdMask(
        const std::vector<data::TensorView> &outputs,
        TaskOutput *result) {

            if (spec_.output_specs.empty()) {
                return core::Status(
                    core::StatusCode::kFailedPrecondition,
                    "No output specification defined");
            }

            if (outputs.empty()) {
                return core::Status(
                    core::StatusCode::kInvalidArgument,
                    "No output tensors from model");
            }
        
            const auto& mask_tensor = outputs[0]; //Assumption is mask_tensors is always the first output
            const auto& shape = mask_tensor.shape();

            if (shape.rank() != 2 && shape.rank() != 3) {
                return core::Status(
                    core::StatusCode::kInvalidArgument,
                    "Expected 2D (H,W) or 3D (B,H,W) mask tensor");
            }

            int height = shape.dim(0);
            int width = shape.rank() == 2 ? shape.dim(1) : shape.dim(2);

            auto seg_mask = std::make_unique<SegmentationMask>();
            seg_mask->height = height;
            seg_mask->width = width;
            seg_mask->class_names = class_labels_;
            seg_mask->dtype = core::DataType::kUint8;

            size_t num_pixels = static_cast<size_t>(height * width);
            seg_mask->mask.resize(num_pixels);
            
            if (mask_tensor.dtype() == core::DataType::kUint8) {
                const uint8_t* data = static_cast<const uint8_t*>(mask_tensor.buffer().data());
                std::copy(data, data + num_pixels, seg_mask->mask.begin());
            } else if (mask_tensor.dtype() == core::DataType::kInt32) {
                const int32_t* data = static_cast<const int32_t*>(mask_tensor.buffer().data());
                for (size_t i = 0; i < num_pixels; ++i) {
                    seg_mask->mask[i] = static_cast<uint8_t>(data[i]);
                }
            } else {
                return core::Status(
                    core::StatusCode::kInvalidArgument,
                    "Unsupported mask tensor dtype for class IDs");
            }
        
            result->segmentation = std::move(seg_mask);
            return core::Status::Ok();
        }

        core::Status SegmentationContract::ParseLogitMask(const std::vector<data::TensorView> &outputs, TaskOutput *result) {

            const auto& logits_tensor = outputs[0];
            const auto& shape = logits_tensor.shape();

            if (shape.rank() != 3) {
                return core::Status(core::StatusCode::kInvalidArgument,
                    "Expected 3D (B,H,W,C) logits tensor");
            }

            std::vector<uint8_t> class_ids;
            core::Status status = ConvertLogitsToClassIds(logits_tensor, &class_ids);
            if (!status.ok()) {
                return status;
            }

            int height = shape.dim(1);
            int width = shape.dim(2);

            auto seg_mask = std::make_unique<SegmentationMask>();
            seg_mask->height = height;
            seg_mask->width = width;
            seg_mask->mask = std::move(class_ids);
            seg_mask->class_names = class_labels_;
            seg_mask->dtype = core::DataType::kUint8;

            result->segmentation = std::move(seg_mask);
            return core::Status::Ok();
        }

        core::Status SegmentationContract::ConvertLogitsToClassIds(const data::TensorView &logits, std::vector<uint8_t> *class_ids) const {

            const auto& shape = logits.shape();
            int height = shape.dim(0);
            int width = shape.dim(1);
            int num_classes = shape.dim(2);

            size_t num_pixels = static_cast<size_t>(height * width);
            class_ids->resize(num_pixels);

            const float* logits_data = static_cast<const float*>(logits.buffer().data());

            for (int y = 0; y<height; ++y) {
                for (int x = 0; x<width; ++x) {
                    int max_class = 0;
                    float max_logit = logits_data[y * width * num_classes + x * num_classes + 0];
                    
                    for (int c = 1; c<num_classes; ++c) {
                        float logit = logits_data[y * width * num_classes + x * num_classes + c];
                        if (logit > max_logit) {
                            max_logit = logit;
                            max_class = c;
                        }
                    }
                    (*class_ids)[y * width + x] = static_cast<uint8_t>(max_class);
                }
            }

            return core::Status::Ok();
        }

        core::Status SegmentationContract::PreProcess(const TaskInput &input, std::vector<data::TensorView> *raw_inputs) {

            raw_inputs->push_back(input.frame.image);
            return core::Status::Ok();
        }

        core::Status SegmentationContract::ValidateOutput(const TaskOutput &output) const {

            for (const auto& invariant : spec_.invariants) {
                core::Status status = invariant.validator(output);
                if (!status.ok()) {
                    return status;
                }
            }
            return ValidateMaskInvariants(*output.segmentation);
        }

        core::Status SegmentationContract::ValidateMaskInvariants(const SegmentationMask &mask) const {

            if (mask.height <= 0 || mask.width <= 0) {
                return core::Status(
                    core::StatusCode::kFailedPrecondition,
                    "Invalid mask dimensions");
            }
            
            size_t expected_size = static_cast<size_t>(mask.height * mask.width);
            if (mask.mask.size() != expected_size) {
                return core::Status(
                    core::StatusCode::kFailedPrecondition,
                    "Mask data size doesn't match dimensions");
            }
            
            return core::Status::Ok();
        }

        core::Status SegmentationContract::ValidateModel(perception::Engine &engine) const {

            auto input_names = engine.InputNames();
            auto output_names = engine.OutputNames();
            
            if (input_names.empty() || output_names.empty()) {
                return core::Status(
                    core::StatusCode::kFailedPrecondition,
                    "Model has no inputs or outputs");
            }
            
            return core::Status::Ok();
        }

}

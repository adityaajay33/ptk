#include "tasks/detection_contract.h"
#include "runtime/core/logger.h"
#include <algorithm>
#include <cmath>

namespace ptk::tasks {

    ObjectDetectionContract::ObjectDetectionContract(const std::vector<std::string>& class_labels, CoordinateSystem output_coords) 
        : class_labels_(class_labels), output_coords_(output_coords) {

        spec_.name = "ObjectDetection";
        spec_.description = "Detect objects with bounding boxes and class labels";

        TaskSpec::InputSpec input_spec;
        input_spec.name = "image";
        input_spec.shape = {-1, -1, 3};
        input_spec.dtype = core::DataType::kFloat32;
        input_spec.layout = core::TensorLayout::kHwc;
        input_spec.allow_batch = true;
        spec_.input_specs.push_back(input_spec);

        TaskSpec::OutputSpec bbox_spec;
        bbox_spec.name = "bounding_boxes";
        bbox_spec.semantic_meaning = "bounding boxes in XYXY format";
        bbox_spec.coordinate_system = output_coords_;
        bbox_spec.dtype = core::DataType::kFloat32;
        spec_.output_specs.push_back(bbox_spec);

        TaskSpec::OutputSpec class_spec;
        class_spec.name = "class_ids";
        class_spec.semantic_meaning = "class IDs for each detection";
        class_spec.dtype = core::DataType::kInt32;
        spec_.output_specs.push_back(class_spec);

        //TaskSpec::OutputSpec confidence_spec;
    }

    core::Status ObjectDetectionContract::ValidateInput(const TaskInput& input) const {
        if (input.frame.image.shape().num_elements() == 0) {
             return core::Status(core::StatusCode::kInvalidArgument, "Input image is empty");
        }
        return core::Status::Ok();
    }

    core::Status ObjectDetectionContract::ValidateOutput(const TaskOutput& output) const {
        if (!output.success) {
            return core::Status(core::StatusCode::kInternal, "Task failed or marked unsuccessful");
        }
        return core::Status::Ok();
    }
    
    core::Status ObjectDetectionContract::ValidateModel(perception::Engine& engine) const {
        return core::Status::Ok();
    }

    core::Status ObjectDetectionContract::PreProcess(const TaskInput& input, std::vector<data::TensorView>* raw_inputs) {
        if (!raw_inputs) return core::Status(core::StatusCode::kInvalidArgument, "raw_inputs is null");
        raw_inputs->clear();
        raw_inputs->push_back(input.frame.image);
        return core::Status::Ok();
    }

    core::Status ObjectDetectionContract::Execute(perception::Engine& engine, const TaskInput& input, TaskOutput& output) {
        auto status = ValidateInput(input);
        if (!status.ok()) return status;
        
        std::vector<data::TensorView> raw_inputs;
        status = PreProcess(input, &raw_inputs);
        if (!status.ok()) return status;
        
        std::vector<data::TensorView> raw_outputs;
        auto engine_status = engine.Infer(raw_inputs, raw_outputs);
        if (!engine_status.ok()) {
            return core::Status(engine_status.code(), engine_status.message());
        }
        
        status = PostProcess(raw_outputs, input, &output);
        if (!status.ok()) return status;
        
        return ValidateOutput(output);
    }
    
    core::Status ObjectDetectionContract::PostProcess(const std::vector<data::TensorView>& raw_outputs, const TaskInput& input, TaskOutput* result) {
         if (raw_outputs.empty()) return core::Status(core::StatusCode::kInternal, "No model output");
         
         // Basic dispatch logic logic - defaulting to YOLO parsing for this fix
         return ParseYOLOOutput(raw_outputs, result);
    }

    core::Status ObjectDetectionContract::ParseYOLOOutput(const std::vector<data::TensorView>& outputs, TaskOutput* result) {
        // Placeholder implementation to satisfy linker
        // TODO: Implement specific YOLO parsing logic matching the model output layout
        return core::Status::Ok();
    }

    core::Status ObjectDetectionContract::ParseFasterRCNNOutput(const std::vector<data::TensorView>& outputs, TaskOutput* result) {
         return core::Status::Ok();
    }

    core::Status ObjectDetectionContract::ValidateDetectionInvariants(const Detection& det) const {
        if (det.confidence < 0.0f || det.confidence > 1.0f) return core::Status(core::StatusCode::kInternal, "Invalid confidence");
        return core::Status::Ok();
    }
    
    core::Status ObjectDetectionContract::ApplyNMS(std::vector<Detection>* detections, float iou_threshold) const {
        // Placeholder NMS
        return core::Status::Ok();
    }
    
    core::Status ObjectDetectionContract::FilterByConfidence(std::vector<Detection>* detections, float threshold) const {
        if(!detections) return core::Status::Ok();
        auto it = std::remove_if(detections->begin(), detections->end(), [threshold](const Detection& d){
            return d.confidence < threshold;
        });
        detections->erase(it, detections->end());
        return core::Status::Ok();
    }

}
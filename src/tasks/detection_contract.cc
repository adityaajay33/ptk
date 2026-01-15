#include "tasks/detection_contract.h"
#include "runtime/core/logger.h"
#include <algorithm>
#include <cmath>

namespace ptk::tasks {

    ObjectDetectionContract::ObjectDetectionContract(const std::vector<std::string>& class_labels, CoordinateSystem output_coords) : class_labels_(class_labels), output_coords_(output_coords) {

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

        TaskSpec::OutputSpec confidence_spec;
    }
}
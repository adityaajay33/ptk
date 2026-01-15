#pragma once

#include "runtime/core/types.h"
#include "runtime/data/frame.h"
#include "task_contract.h"
#include <vector>
#include <string>
#include <memory>

namespace ptk::tasks {

    struct BoundingBox {
        float x1, y1, x2, y2, score;
        CoordinateSystem coordinate_system;

        float Width() const { return x2 - x1; }
        float Height() const { return y2 - y1; }
        float Area() const { return Width() * Height(); }
        float CenterX() const { return (x1 + x2) / 2.0f; }
        float CenterY() const { return (y1 + y2) / 2.0f; }
    };

    struct Detection {
        BoundingBox box;
        int class_id;
        std::string class_name;
        float confidence;
        std::vector<float> embeddings;

        int64_t timestamp_ns = 0;
        int track_id = -1;
    };

    struct Keypoint {
        float x, y;
        float confidence;
        int keypoint_id;
        std::string keypoint_name;
        bool is_visible;
        CoordinateSystem coordinate_system;
    };

    struct SegmentationMask {

        std::vector<uint8_t> mask;
        int height;
        int width;
        std::vector<std::string> class_names;
        core::DataType dtype;
    };

    struct TaskOutput {
        std::string task_type;
        bool success = false;
        
        std::vector<Detection> detections;
        std::unique_ptr<SegmentationMask> segmentation;
        std::vector<float> classification_scores;
        std::vector<float> embedding;
        
        // Metadata
        int64_t timestamp_ns = 0;
        int64_t frame_index = 0;
        float inference_time_ms = 0.0f;
        
        int original_width = 0;
        int original_height = 0;
        
        core::Status ConvertCoordinates(
            CoordinateSystem from,
            CoordinateSystem to,
            int image_width,
            int image_height);
    };

    struct TaskInput {
        data::Frame frame;
        
        // Task-specific parameters
        struct Parameters {
            float confidence_threshold = 0.5f;
            float nms_threshold = 0.45f;
            int max_detections = 100;
            std::vector<int> class_filter;  // empty = all classes
        };
        Parameters params;
    };

}
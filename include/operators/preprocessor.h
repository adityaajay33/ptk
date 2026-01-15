#pragma once

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "runtime/data/frame_msg.h"
#include "runtime/core/types.h"
#include "operators/normalization_params.h"
#include <rclcpp/rclcpp.hpp>

namespace ptk {

    struct PreprocessorConfig {
        core::TensorLayout input_layout;
        core::PixelFormat input_format;
        core::DataType input_type;

        core::TensorLayout output_layout;
        core::PixelFormat output_format;
        core::DataType output_type;

        bool convert_rgb_to_bgr;
        bool normalize;
        bool add_batch_dimension;
        bool to_grayscale;
        operators::NormalizationParams norm;

        int target_height;
        int target_width;
    };

    class Preprocessor : public components::ComponentInterface {
        public:
            explicit Preprocessor(
                const PreprocessorConfig& config,
                const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
            ~Preprocessor() override = default;

            void BindInput(core::InputPort<data::Frame>* in);
            void BindOutput(core::OutputPort<data::Frame>* out);

            core::Status Init(core::RuntimeContext* context) override;
            core::Status Start() override;
            core::Status Stop() override;
            void Tick() override;

        private:
            void FrameCallback(std::unique_ptr<data::FrameMsg> msg);
            std::unique_ptr<data::FrameMsg> ProcessFrame(std::unique_ptr<data::FrameMsg> input);
            
            core::RuntimeContext* context_;
            core::InputPort<data::Frame>* input_;
            core::OutputPort<data::Frame>* output_;
            PreprocessorConfig config_;

            std::vector<float> float_buffer_;
            std::vector<std::uint8_t> uint8_temp_;

            data::Frame output_frame_;
            
            rclcpp::Subscription<data::FrameMsg>::SharedPtr frame_subscription_;
            rclcpp::Publisher<data::FrameMsg>::SharedPtr processed_publisher_;
    };
}


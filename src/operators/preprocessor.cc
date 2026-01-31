#include "operators/preprocessor.h"
#include "operators/cast_uint8_to_float32.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/core/status.h"
#include <mutex>

namespace ptk
{

    Preprocessor::Preprocessor(const rclcpp::NodeOptions &options)
        : ComponentInterface("preprocessor", options),
          context_(nullptr),
          input_(nullptr),
          output_(nullptr),
          config_(),
          float_buffer_(),
          uint8_temp_(),
          output_frame_()
    {

        this->declare_parameter("target_height", 224);
        this->declare_parameter("target_width", 224);
        this->declare_parameter("normalize", true);
        this->declare_parameter("add_batch_dimension", false);
        this->declare_parameter("to_grayscale", false);
        this->declare_parameter("convert_rgb_to_bgr", false);
        
        config_.target_height = this->get_parameter("target_height").as_int();
        config_.target_width = this->get_parameter("target_width").as_int();
        config_.normalize = this->get_parameter("normalize").as_bool();
        config_.add_batch_dimension = this->get_parameter("add_batch_dimension").as_bool();
        config_.to_grayscale = this->get_parameter("to_grayscale").as_bool();
        config_.convert_rgb_to_bgr = this->get_parameter("convert_rgb_to_bgr").as_bool();
        
        // Set reasonable defaults for other config fields
        config_.input_layout = core::TensorLayout::kHwc;
        config_.input_format = core::PixelFormat::kRgb8;
        config_.input_type = core::DataType::kUint8;
        config_.output_layout = core::TensorLayout::kHwc;
        config_.output_format = core::PixelFormat::kRgb8;
        config_.output_type = core::DataType::kFloat32;
        
        // Default normalization params (ImageNet)
        config_.norm.mean[0] = 0.485f;
        config_.norm.mean[1] = 0.456f;
        config_.norm.mean[2] = 0.406f;
        config_.norm.std[0] = 0.229f;
        config_.norm.std[1] = 0.224f;
        config_.norm.std[2] = 0.225f;
        config_.norm.num_channels = 3;
    }

    void Preprocessor::BindInput(core::InputPort<data::Frame>* in)
    {
        input_ = in;
    }

    void Preprocessor::BindOutput(core::OutputPort<data::Frame>* out)
    {
        output_ = out;
    }

    core::Status Preprocessor::Init(core::RuntimeContext *context)
    {
        if (context == nullptr)
        {
            return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
        }
        context_ = context;
        return core::Status::Ok();
    }

    core::Status Preprocessor::Start()
    {
        if (input_ == nullptr || !input_->is_bound() ||
            output_ == nullptr || !output_->is_bound())
        {
            return core::Status(
                core::StatusCode::kFailedPrecondition,
                "Preprocessor ports not bound");
        }
        return core::Status::Ok();
    }

    core::Status Preprocessor::Stop()
    {
        return core::Status::Ok();
    }

    void Preprocessor::Tick()
    {
        if (context_ == nullptr)
        {
            return;
        }
        if (input_ == nullptr || !input_->is_bound())
        {
            context_->LogError("Preprocessor: input port not bound");
            return;
        }
        if (output_ == nullptr || !output_->is_bound())
        {
            context_->LogError("Preprocessor: output port not bound");
            return;
        }

        auto frame_opt = input_->Pop(std::chrono::milliseconds(10));
        if (!frame_opt)
        {
            return;
        }

        const data::Frame &in = *frame_opt;
        
        // Ensure output frame is allocated if needed
        if (output_frame_.image.empty()) {
            output_frame_ = data::Frame::CreateOwned(
                config_.target_height, config_.target_width, 3,
                config_.output_format, config_.output_layout);
        }

        data::Frame &out = output_frame_;

        out.frame_index = in.frame_index;
        out.timestamp_ns = in.timestamp_ns;
        out.camera_id = in.camera_id;
        out.pixel_format = in.pixel_format;

        const data::TensorView &src = in.image;
        data::TensorView &dst = out.image;

        core::Status s = operators::CastUint8ToFloat32(src, &dst);
        if (!s.ok())
        {
            context_->LogError("Preprocessor: CastUint8ToFloat32 failed");
            return;
        }

        output_->Push(std::move(out));
        // Reset output_frame_ for next tick since it was moved
        output_frame_ = data::Frame();
    }

} // namespace ptk

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::Preprocessor)

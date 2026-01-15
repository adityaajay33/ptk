#include "operators/preprocessor.h"
#include "operators/cast_uint8_to_float32.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/status.h"

namespace ptk
{

    Preprocessor::Preprocessor(const PreprocessorConfig &config, const rclcpp::NodeOptions &options)
        : ComponentInterface("preprocessor", options),
          context_(nullptr),
          input_(nullptr),
          output_(nullptr),
          config_(config),
          float_buffer_(),
          uint8_temp_(),
          output_frame_()
    {
        // Create subscriber for zero-copy frame reception
        frame_subscription_ = this->create_subscription<data::FrameMsg>(
            "camera/frames",
            rclcpp::QoS(10).best_effort(),
            [this](std::unique_ptr<data::FrameMsg> msg) {
                this->FrameCallback(std::move(msg));
            }
        );
        
        // Create publisher for zero-copy processed frame output
        processed_publisher_ = this->create_publisher<data::FrameMsg>(
            "preprocessor/frames",
            rclcpp::QoS(10).best_effort()
        );
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

        const data::Frame *in = input_->get();
        data::Frame *out = output_->get();

        if (in == nullptr || out == nullptr)
        {
            context_->LogError("Preprocessor: null frame from port");
            return;
        }

        // Copy basic metadata.
        out->frame_index = in->frame_index;
        out->timestamp_ns = in->timestamp_ns;
        out->camera_id = in->camera_id;
        out->pixel_format = in->pixel_format;

        // Your Frame::image is a plain TensorView, not optional.
        // Use TensorView::empty() to check validity.
        if (in->image.empty())
        {
            context_->LogError("Preprocessor: input frame has empty image tensor");
            return;
        }
        if (out->image.empty())
        {
            context_->LogError("Preprocessor: output frame has empty image tensor");
            return;
        }

        const data::TensorView &src = in->image;
        data::TensorView &dst = out->image;

        // For now: simple uint8 -> float32 cast.
        core::Status s = operators::CastUint8ToFloat32(src, &dst);
        if (!s.ok())
        {
            context_->LogError("Preprocessor: CastUint8ToFloat32 failed");
            return;
        }
    }

    void Preprocessor::FrameCallback(std::unique_ptr<data::FrameMsg> msg)
    {

        auto processed = ProcessFrame(std::move(msg));
        
        if (processed) {
            processed_publisher_->publish(std::move(processed));
        }
    }

    std::unique_ptr<data::FrameMsg> Preprocessor::ProcessFrame(std::unique_ptr<data::FrameMsg> input)
    {
        if (!input || input->buffer_data.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Preprocessor: received empty frame");
            return nullptr;
        }

        const data::TensorView src = input->GetTensorView();
        
        size_t num_elements = src.shape().num_elements();
        std::vector<uint8_t> float_data(num_elements * sizeof(float));
        
        data::BufferView dst_buffer(
            float_data.data(),
            float_data.size(),
            core::DeviceType::kCpu
        );
        
        data::TensorView dst(dst_buffer, core::DataType::kFloat32, src.shape());
        
        core::Status s = operators::CastUint8ToFloat32(src, &dst);
        if (!s.ok()) {
            RCLCPP_ERROR(this->get_logger(), "Preprocessor: CastUint8ToFloat32 failed");
            return nullptr;
        }

        auto output = data::FrameMsg::Create(
            std::move(float_data),
            core::DataType::kFloat32,
            src.shape(),
            input->pixel_format,
            input->layout,
            input->timestamp_ns,
            input->frame_index,
            input->camera_id
        );
        
        return output;
    }

} // namespace ptk

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::Preprocessor)

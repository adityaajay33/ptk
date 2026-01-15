#include "sensors/mac_camera.h"
#include <opencv2/opencv.hpp>
#include <cstring>

namespace ptk::sensors
{

        struct MacCamera::Impl
        {
            cv::VideoCapture cap;
        };

        MacCamera::MacCamera(int device_index, const rclcpp::NodeOptions &options)
            : CameraInterface("mac_camera", options),
              device_index_(device_index),
              is_running_(false),
              frame_index_(0),
              impl_(new Impl())
        {
            //create publisher for zero-copy frame transport
            frame_publisher_ = this->create_publisher<data::FrameMsg>(
                "camera/frames",
                rclcpp::QoS(10).best_effort()  
            );
        }

        MacCamera::~MacCamera()
        {
            Stop();
            delete impl_;
        }

        core::Status MacCamera::Init()
        {
            if (device_index_ < 0)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                                    "MacCamera: invalid device index");
            }
            return core::Status::Ok();
        }

        core::Status MacCamera::Start()
        {
            if (is_running_)
            {
                return core::Status::Ok();
            }

            if (!impl_->cap.open(device_index_))
            {
                return core::Status(core::StatusCode::kInternal,
                                    "MacCamera: failed to open camera device");
            }

            is_running_ = true;
            return core::Status::Ok();
        }

        core::Status MacCamera::Stop()
        {
            if (!is_running_)
            {
                return core::Status::Ok();
            }

            impl_->cap.release();
            is_running_ = false;

            return core::Status::Ok();
        }

        core::Status MacCamera::GetFrame(ptk::data::Frame *out)
        {
            if (!is_running_)
            {
                return core::Status(core::StatusCode::kFailedPrecondition,
                                    "MacCamera: GetFrame() while camera not running");
            }

            if (out == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                                    "MacCamera: out == nullptr");
            }

            cv::Mat img;
            if (!impl_->cap.read(img))
            {
                return core::Status(core::StatusCode::kInternal,
                                    "MacCamera: failed to read frame");
            }

            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            int H = img.rows;
            int W = img.cols;
            int C = img.channels();

            size_t num_bytes = static_cast<size_t>(H * W * C);

            // Allocate buffer
            std::vector<uint8_t> buffer(num_bytes);
            std::memcpy(buffer.data(), img.data, num_bytes);

            // Fill Frame
            out->image = ptk::data::TensorView(
                ptk::data::BufferView(buffer.data(), num_bytes, core::DeviceType::kCpu),
                core::DataType::kUint8,
                ptk::data::TensorShape({H, W, C}));

            out->pixel_format = core::PixelFormat::kRgb8;
            out->layout = core::TensorLayout::kHwc;
            out->frame_index = frame_index_++;
            out->timestamp_ns = 0;
            out->camera_id = device_index_;

            return core::Status::Ok();
        }

        void MacCamera::Tick()
        {
            PublishFrame();
        }
        
        void MacCamera::PublishFrame()
        {
            if (!is_running_)
            {
                return;
            }

            cv::Mat img;
            if (!impl_->cap.read(img))
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to read frame from camera");
                return;
            }

            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            int H = img.rows;
            int W = img.cols;
            int C = img.channels();
            size_t num_bytes = static_cast<size_t>(H * W * C);

            std::vector<uint8_t> buffer(num_bytes);
            std::memcpy(buffer.data(), img.data, num_bytes);

            // Create FrameMsg with owned data (zero-copy via unique_ptr)
            auto frame_msg = data::FrameMsg::Create(
                std::move(buffer),
                core::DataType::kUint8,
                data::TensorShape({H, W, C}),
                core::PixelFormat::kRgb8,
                core::TensorLayout::kHwc,
                this->now().nanoseconds(),
                frame_index_++,
                device_index_
            );

            // Publish with zero-copy (ownership transfer)
            frame_publisher_->publish(std::move(frame_msg));
            
            RCLCPP_DEBUG(this->get_logger(), "Published frame %d", frame_index_ - 1);
        }

} // namespace ptk::sensors

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::sensors::MacCamera)
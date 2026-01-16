#include "sensors/mac_camera.h"
#include <opencv2/opencv.hpp>
#include <cstring>

namespace ptk::sensors
{

        struct MacCamera::Impl
        {
            cv::VideoCapture cap;
        };

        MacCamera::MacCamera(const rclcpp::NodeOptions &options)
            : CameraInterface("mac_camera", options),
              device_index_(0),
              is_running_(false),
              frame_index_(0),
              impl_(new Impl())
        {
            // Load device index from ROS parameter
            this->declare_parameter("device_index", 0);
            device_index_ = this->get_parameter("device_index").as_int();
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
            // Tick method for scheduler integration
            // In a real implementation, this could trigger frame capture
            // or other periodic operations
        }

} // namespace ptk::sensors

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::sensors::MacCamera)

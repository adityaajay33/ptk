#include "sensors/camera.h"
#include <opencv2/opencv.hpp>
#include <cstring>

namespace ptk::sensors
{

        struct Camera::Impl
        {
            cv::VideoCapture cap;
        };

        Camera::Camera(const rclcpp::NodeOptions &options)
            : CameraInterface("camera", options),
              device_index_(0),
              is_running_(false),
              frame_index_(0),
              impl_(new Impl()),
              current_buffer_index_(0),
              output_(nullptr)
        {
            this->declare_parameter("device_index", 0);
            device_index_ = this->get_parameter("device_index").as_int();
        }

        Camera::~Camera()
        {
            Stop();
            delete impl_;
        }

        core::Status Camera::Init()
        {
            if (device_index_ < 0)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                                    "Camera: invalid device index");
            }
            return core::Status::Ok();
        }

        core::Status Camera::Start()
        {
            if (is_running_)
            {
                return core::Status::Ok();
            }

            if (!impl_->cap.open(device_index_))
            {
                return core::Status(core::StatusCode::kInternal,
                                    "Camera: failed to open camera device");
            }

            is_running_ = true;
            return core::Status::Ok();
        }

        core::Status Camera::Stop()
        {
            if (!is_running_)
            {
                return core::Status::Ok();
            }

            impl_->cap.release();
            is_running_ = false;

            return core::Status::Ok();
        }

        core::Status Camera::GetFrame(ptk::data::Frame *out)
        {
            if (!is_running_)
            {
                return core::Status(core::StatusCode::kFailedPrecondition,
                                    "Camera: GetFrame() while camera not running");
            }

            if (out == nullptr)
            {
                return core::Status(core::StatusCode::kInvalidArgument,
                                    "Camera: out == nullptr");
            }

            cv::Mat img;
            if (!impl_->cap.read(img))
            {
                return core::Status(core::StatusCode::kInternal,
                                    "Camera: failed to read frame");
            }

            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            int H = img.rows;
            int W = img.cols;
            int C = img.channels();

            size_t num_bytes = static_cast<size_t>(H * W * C);

            std::vector<uint8_t> buffer(num_bytes);
            std::memcpy(buffer.data(), img.data, num_bytes);

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

        void Camera::BindOutput(core::OutputPort<data::Frame>* out)
        {
            output_ = out;
        }

        void Camera::Tick()
        {
            if (!is_running_ || output_ == nullptr || !output_->is_bound()) {
                return;
            }

            data::Frame* out = output_->get();
            if (out == nullptr) {
                return;
            }

            cv::Mat img;
            if (!impl_->cap.read(img)) {
                return;
            }

            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            int H = img.rows;
            int W = img.cols;
            int C = img.channels();
            size_t num_bytes = static_cast<size_t>(H * W * C);

            current_buffer_index_ = 1 - current_buffer_index_;
            std::vector<uint8_t>& active_buffer = frame_buffer_[current_buffer_index_];

            std::unique_lock<std::mutex> lock(output_mutex_);

            active_buffer.resize(num_bytes);
            std::memcpy(active_buffer.data(), img.data, num_bytes);

            out->image = ptk::data::TensorView(
                ptk::data::BufferView(active_buffer.data(), num_bytes, core::DeviceType::kCpu),
                core::DataType::kUint8,
                ptk::data::TensorShape({H, W, C})
            );

            out->pixel_format = core::PixelFormat::kRgb8;
            out->layout = core::TensorLayout::kHwc;
            out->frame_index = frame_index_++;
            out->timestamp_ns = 0;
            out->camera_id = device_index_;
        }

} // namespace ptk::sensors

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ptk::sensors::Camera)

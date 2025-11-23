#include "sensors/mac_camera.h"

#include <opencv2/opencv.hpp>
#include <cstring> // memcpy

namespace ptk
{
    namespace sensors
    {

        struct MacCamera::Impl
        {
            cv::VideoCapture cap;
        };

        MacCamera::MacCamera(int device_index)
            : impl_(new Impl()),
              device_index_(device_index),
              is_running_(false),
              frame_index_(0) {}

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
                                    "MacCamera: GetFrame while camera not running");
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

            // Convert BGR -> RGB
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            int H = img.rows;
            int W = img.cols;
            int C = img.channels();
            std::size_t num_bytes = static_cast<std::size_t>(H * W * C);

            // Allocate memory manually because you have no Buffer class
            uint8_t *mem = new uint8_t[num_bytes];
            std::memcpy(mem, img.data, num_bytes);

            // Wrap memory into BufferView
            ptk::data::BufferView buffer_view(mem, num_bytes, core::DeviceType::kCpu);

            // Create tensor shape
            ptk::data::TensorShape shape({H, W, C});

            // Assign into Frame::image
            out->image = ptk::data::TensorView(
                buffer_view,
                core::DataType::kUint8,
                shape);

            // Fill metadata
            out->pixel_format = core::PixelFormat::kRgb8;
            out->layout = core::TensorLayout::kHwc;
            out->timestamp_ns = 0; // TODO: Integrate runtime clock
            out->frame_index = frame_index_++;
            out->camera_id = device_index_;

            return core::Status::Ok();
        }

        void MacCamera::Tick() {
            if (!output_port_.is_bound()) {
                return;
            }

            ptk::data::Frame frame;
            core::Status s = GetFrame(&frame);
            if (!s.ok()) {
                std::fprintf(stderr, "MacCamera Tick failed: %s\n", s.message().c_str());
                return;
            }

            *(output_port_.get()) = frame;
        }

    } // namespace sensors
} // namespace ptk
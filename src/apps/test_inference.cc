#include <rclcpp/rclcpp.hpp>
#include "runtime/components/inference_node.h"
#include "runtime/components/synthetic_camera.h"
#include "operators/preprocessor.h"
#include "runtime/core/runtime_context.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"
#include "tasks/task_output.h"
#include <fstream>

using namespace ptk;

/**
 * Simple component that reads TaskOutput and writes detection results to a file
 */
class DetectionWriter : public components::ComponentInterface
{
public:
    explicit DetectionWriter(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : ComponentInterface("detection_writer", options),
          context_(nullptr),
          input_(nullptr),
          output_file_("detections_output.txt")
    {
    }

    void BindInput(core::InputPort<tasks::TaskOutput> *in)
    {
        input_ = in;
    }

    core::Status Init(core::RuntimeContext *context) override
    {
        context_ = context;
        file_.open(output_file_);
        if (!file_.is_open())
        {
            return core::Status(core::StatusCode::kInternal,
                                "Failed to open output file: " + output_file_);
        }

        file_ << "=== Detection Results ===" << std::endl;
        file_ << "Format: frame_idx | num_detections | inference_time_ms | [class confidence x1 y1 x2 y2]" << std::endl;
        file_ << std::endl;

        RCLCPP_INFO(this->get_logger(), "DetectionWriter initialized, writing to: %s",
                    output_file_.c_str());
        return core::Status::Ok();
    }

    core::Status Start() override
    {
        RCLCPP_INFO(this->get_logger(), "DetectionWriter started");
        return core::Status::Ok();
    }

    core::Status Stop() override
    {
        if (file_.is_open())
        {
            file_.close();
        }
        RCLCPP_INFO(this->get_logger(), "DetectionWriter stopped. Results saved to: %s",
                    output_file_.c_str());
        return core::Status::Ok();
    }

    void Tick() override
    {
        if (!input_ || !input_->is_bound())
        {
            return;
        }

        const tasks::TaskOutput *result = input_->get();

        if (!result->success)
        {
            RCLCPP_WARN(this->get_logger(), "Received failed inference result");
            return;
        }

        // Write frame info
        file_ << "Frame " << result->frame_index
              << " | Detections: " << result->detections.size()
              << " | Time: " << result->inference_time_ms << " ms"
              << std::endl;

        // Write each detection
        for (const auto &det : result->detections)
        {
            file_ << "  " << det.class_name
                  << " (conf=" << det.confidence << ")"
                  << " [" << det.box.x1 << ", " << det.box.y1
                  << ", " << det.box.x2 << ", " << det.box.y2 << "]"
                  << std::endl;

            // Log to console
            RCLCPP_INFO(this->get_logger(),
                        "  Detected: %s (%.2f) at [%.1f, %.1f, %.1f, %.1f]",
                        det.class_name.c_str(), det.confidence,
                        det.box.x1, det.box.y1, det.box.x2, det.box.y2);
        }

        file_ << std::endl;
        file_.flush();
    }

private:
    core::RuntimeContext *context_;
    core::InputPort<tasks::TaskOutput> *input_;
    std::string output_file_;
    std::ofstream file_;
};

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);

    std::cout << "=== PTK Inference Pipeline Test ===" << std::endl;
    std::cout << "Pipeline: SyntheticCamera -> Preprocessor -> InferenceNode -> DetectionWriter" << std::endl;
    std::cout << std::endl;

    // Create runtime context
    auto context = std::make_shared<core::RuntimeContext>();

    // Create components
    auto camera = std::make_shared<components::SyntheticCamera>(rclcpp::NodeOptions());
    auto preprocessor = std::make_shared<Preprocessor>(rclcpp::NodeOptions());
    auto inference = std::make_shared<components::InferenceNode>(rclcpp::NodeOptions());
    auto writer = std::make_shared<DetectionWriter>(rclcpp::NodeOptions());

    // Create ports for zero-copy communication
    data::Frame camera_frame;
    data::Frame preprocessed_frame;
    tasks::TaskOutput inference_output;

    core::OutputPort<data::Frame> camera_out;
    camera_out.Bind(&camera_frame);

    core::InputPort<data::Frame> preproc_in;
    preproc_in.Bind(&camera_frame);

    core::OutputPort<data::Frame> preproc_out;
    preproc_out.Bind(&preprocessed_frame);

    core::InputPort<data::Frame> inference_in;
    inference_in.Bind(&preprocessed_frame);

    core::OutputPort<tasks::TaskOutput> inference_out;
    inference_out.Bind(&inference_output);

    core::InputPort<tasks::TaskOutput> writer_in;
    writer_in.Bind(&inference_output);

    // Bind ports to components
    camera->BindOutput(&camera_out);
    preprocessor->BindInput(&preproc_in);
    preprocessor->BindOutput(&preproc_out);
    inference->BindInput(&inference_in);
    inference->BindOutput(&inference_out);
    writer->BindInput(&writer_in);

    // Initialize components
    std::cout << "Initializing components..." << std::endl;

    auto status = camera->Init(context.get());
    if (!status.ok())
    {
        std::cerr << "Failed to init camera: " << status.message() << std::endl;
        return 1;
    }

    status = preprocessor->Init(context.get());
    if (!status.ok())
    {
        std::cerr << "Failed to init preprocessor: " << status.message() << std::endl;
        return 1;
    }

    status = inference->Init(context.get());
    if (!status.ok())
    {
        std::cerr << "Failed to init inference: " << status.message() << std::endl;
        return 1;
    }

    status = writer->Init(context.get());
    if (!status.ok())
    {
        std::cerr << "Failed to init writer: " << status.message() << std::endl;
        return 1;
    }

    // Start components
    camera->Start();
    preprocessor->Start();
    inference->Start();
    writer->Start();

    std::cout << "Running inference pipeline..." << std::endl;
    std::cout << "(This will run inference on synthetic frames)" << std::endl;
    std::cout << std::endl;

    // Run pipeline for N frames
    const int num_frames = 10;
    for (int i = 0; i < num_frames; i++)
    {
        camera->Tick();       // Generate synthetic frame
        preprocessor->Tick(); // Preprocess frame
        inference->Tick();    // Run inference
        writer->Tick();       // Write results

        // Small delay to simulate real-time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop components
    camera->Stop();
    preprocessor->Stop();
    inference->Stop();
    writer->Stop();

    std::cout << std::endl;
    std::cout << "Pipeline test complete!" << std::endl;
    std::cout << "Detection results saved to: detections_output.txt" << std::endl;

    rclcpp::shutdown();
    return 0;
}

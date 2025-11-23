// src/components/synthetic_camera.cc
#include "runtime/components/synthetic_camera.h"

#include "runtime/core/runtime_context.h"
#include "runtime/data/frame.h"

namespace ptk {
namespace components {

    SyntheticCamera::SyntheticCamera()
        : context_(nullptr), output_(nullptr), frame_index_(0) {}

    void SyntheticCamera::BindOutput(core::OutputPort<data::Frame>* port) {
    output_ = port;
    }

    core::Status SyntheticCamera::Init(core::RuntimeContext* context) {
        if (context == nullptr) {
            return core::Status(core::StatusCode::kInvalidArgument, "Context is null");
        }
        context_ = context;
        frame_index_ = 0;
        return core::Status::Ok();
    }

    core::Status SyntheticCamera::Start() {
        if (output_ == nullptr || !output_->is_bound()) {
            return core::Status(core::StatusCode::kFailedPrecondition,
                        "SyntheticCamera output not bound");
        }
        context_->LogInfo("SyntheticCamera started.");
        return core::Status::Ok();
    }

    void SyntheticCamera::Stop() {
        context_->LogInfo("SyntheticCamera stopped.");
    }

    void SyntheticCamera::Tick() {
        if (output_ == nullptr || !output_->is_bound()) {
            context_->LogError("SyntheticCamera Tick with unbound output.");
            return;
        }

        data::Frame* frame = output_->get();
        if (frame == nullptr) {
            context_->LogError("SyntheticCamera Tick with null frame.");
            return;
        }

        frame->frame_index = frame_index_++;
        frame->timestamp_ns = context_->NowNanoseconds();
    }

}  // namespace components
}  // namespace ptk
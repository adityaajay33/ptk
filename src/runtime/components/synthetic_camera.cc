// src/components/synthetic_camera.cc
#include "runtime/components/synthetic_camera.h"

#include "runtime/core/runtime_context.h"
#include "runtime/data/frame.h"

namespace ptk {

    SyntheticCamera::SyntheticCamera()
        : context_(nullptr), output_(nullptr), frame_index_(0) {}

    void SyntheticCamera::BindOutput(OutputPort<Frame>* port) {
    output_ = port;
    }

    Status SyntheticCamera::Init(RuntimeContext* context) {
        if (context == nullptr) {
            return Status(StatusCode::kInvalidArgument, "Context is null");
        }
        context_ = context;
        frame_index_ = 0;
        return Status::Ok();
    }

    Status SyntheticCamera::Start() {
        if (output_ == nullptr || !output_->is_bound()) {
            return Status(StatusCode::kFailedPrecondition,
                        "SyntheticCamera output not bound");
        }
        context_->LogInfo("SyntheticCamera started.");
        return Status::Ok();
    }

    void SyntheticCamera::Stop() {
        context_->LogInfo("SyntheticCamera stopped.");
    }

    void SyntheticCamera::Tick() {
        if (output_ == nullptr || !output_->is_bound()) {
            context_->LogError("SyntheticCamera Tick with unbound output.");
            return;
        }

        Frame* frame = output_->get();
        if (frame == nullptr) {
            context_->LogError("SyntheticCamera Tick with null frame.");
            return;
        }

        frame->frame_index = frame_index_++;
        frame->timestamp_ns = context_->NowNanoseconds();
    }

}  // namespace ptk
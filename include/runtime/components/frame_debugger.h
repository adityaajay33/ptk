// include/runtime/components/frame_debugger.h
#ifndef RUNTIME_COMPONENTS_FRAME_DEBUGGER_H_
#define RUNTIME_COMPONENTS_FRAME_DEBUGGER_H_

#include "runtime/components/component_interface.h"
#include "runtime/core/port.h"
#include "runtime/data/frame.h"

namespace ptk {

class FrameDebugger : public ComponentInterface {
 public:
  FrameDebugger();
  ~FrameDebugger() override = default;

  // The pipeline or app calls this to connect a Frame source.
  void BindInput(InputPort<Frame>* port);

  Status Init(RuntimeContext* context) override;
  Status Start() override;
  void Stop() override;
  void Tick() override;

 private:
  RuntimeContext* context_;
  InputPort<Frame>* input_;
  int tick_count_;
};

}  // namespace ptk

#endif  // RUNTIME_COMPONENTS_FRAME_DEBUGGER_H_
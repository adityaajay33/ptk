// demo.cc
#include <cstdio>
#include <cstdint>
#include <vector>

#include "runtime/core/runtime_context.h"
#include "runtime/core/scheduler.h"
#include "runtime/core/port.h"

#include "runtime/data/frame.h"
#include "runtime/data/tensor.h"
#include "runtime/core/types.h"

#include "runtime/components/synthetic_camera.h"
#include "runtime/components/frame_debugger.h"

int main(int argc, char** argv) {
  // 1) Initialize runtime context.
  runtime::RuntimeContext ctx;
  runtime::RuntimeContextOptions opts;
  runtime::Status s = ctx.Init(opts);
  if (!s.ok()) {
    std::fprintf(stderr, "Context init failed: %s\n",
                 s.message().c_str());
    return 1;
  }

  // 2) Allocate pixel storage on CPU.
  const int height = 480;
  const int width = 640;
  const int channels = 3;

  std::vector<std::uint8_t> buffer(
      static_cast<std::size_t>(height) *
      static_cast<std::size_t>(width) *
      static_cast<std::size_t>(channels));

  // 3) Wrap it into BufferView, TensorView, and Frame.
  runtime::BufferView buf_view(
      buffer.data(),
      buffer.size(),
      runtime::DeviceType::kCpu);

  runtime::TensorShape shape({height, width, channels});
  runtime::TensorView image_tensor(buf_view,
                                   runtime::DataType::kUint8,
                                   shape);

  runtime::Frame frame;
  frame.image = image_tensor;
  frame.pixel_format = runtime::PixelFormat::kRgb8;
  frame.timestamp_ns = ctx.NowNanoseconds();
  frame.frame_index = 0;
  frame.camera_id = 0;

  // 4) Create ports and bind them to shared frame storage.
  runtime::OutputPort<runtime::Frame> camera_out;
  runtime::InputPort<runtime::Frame> debug_in;

  camera_out.Bind(&frame);
  debug_in.Bind(&frame);

  // 5) Create components and connect ports.
  runtime::SyntheticCamera camera;
  runtime::FrameDebugger debugger;

  camera.BindOutput(&camera_out);
  debugger.BindInput(&debug_in);

  // 6) Initialize scheduler.
  runtime::Scheduler scheduler;
  s = scheduler.Init(&ctx);
  if (!s.ok()) {
    ctx.LogError("Scheduler init failed.");
    return 1;
  }

  s = scheduler.AddComponent(&camera);
  if (!s.ok()) {
    ctx.LogError("AddComponent(camera) failed.");
    return 1;
  }

  s = scheduler.AddComponent(&debugger);
  if (!s.ok()) {
    ctx.LogError("AddComponent(debugger) failed.");
    return 1;
  }

  // 7) Start scheduler and run a few ticks.
  s = scheduler.Start();
  if (!s.ok()) {
    ctx.LogError("Scheduler start failed.");
    return 1;
  }

  // Run five ticks. SyntheticCamera updates frame metadata,
  // FrameDebugger reads the frame and logs its shape.
  scheduler.RunLoop(5);

  scheduler.Stop();
  ctx.Shutdown();
  return 0;
}
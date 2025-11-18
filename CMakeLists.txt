cmake_minimum_required(VERSION 3.16)
project(runtime CXX)

# Use modern-ish C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Runtime library ----

add_library(runtime
    src/core/runtime_context.cc
    src/core/scheduler.cc
    src/components/counter.cc
    src/components/heartbeat.cc
    src/components/frame_debugger.cc
    src/components/synthetic_camera.cc
    src/components/pipeline.cc
)

target_include_directories(runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(demo
    demo.cc
)

target_link_libraries(demo PRIVATE runtime)
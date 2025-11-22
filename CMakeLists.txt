cmake_minimum_required(VERSION 3.16)
project(runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(runtime
    # Core
    src/runtime/core/runtime_context.cc
    src/runtime/core/scheduler.cc
    src/runtime/core/pipeline.cc

    src/runtime/components/counter.cc
    src/runtime/components/heartbeat.cc
    src/runtime/components/frame_debugger.cc
    src/runtime/components/synthetic_camera.cc

    src/runtime/preprocess/preprocessor.cc
    src/runtime/preprocess/transforms.cc
)

target_include_directories(runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(demo
    src/apps/demo.cc
)

target_link_libraries(demo PRIVATE runtime)

target_include_directories(demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
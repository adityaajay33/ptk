cmake_minimum_required(VERSION 3.16)
project(ptk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(runtime
    # Runtime Core
    src/runtime/core/runtime_context.cc
    src/runtime/core/scheduler.cc
    src/runtime/core/pipeline.cc

    # Runtime Components
    src/runtime/components/counter.cc
    src/runtime/components/heartbeat.cc
    src/runtime/components/frame_debugger.cc
    src/runtime/components/synthetic_camera.cc

    # Operators - Main
    src/operators/preprocessor.cc

    # Operators - Layout Transforms
    src/operators/hwc_to_chw.cc
    src/operators/chw_to_hwc.cc

    # Operators - Type Casting
    src/operators/cast_uint8_to_float32.cc
    src/operators/cast_float32_to_uint8.cc

    # Operators - Color Space
    src/operators/rgb_to_gray.cc
    src/operators/rgb_to_bgr.cc
    src/operators/bgr_to_rgb.cc

    # Operators - Image Processing
    src/operators/normalize.cc
    src/operators/pad_to_size.cc
    src/operators/center_crop.cc
    src/operators/add_batch_dim.cc
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
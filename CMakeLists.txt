cmake_minimum_required(VERSION 3.14)
project(ptk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable intra-process communication for zero-copy
add_definitions(-DRCLCPP_INTRA_PROCESS_COMMS_ENABLED)

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

# Point OpenCV to Conda installation if available
if(DEFINED ENV{CONDA_PREFIX})
    set(OpenCV_DIR "$ENV{CONDA_PREFIX}/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Check for CUDA (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling GPU support")
    add_definitions(-DPTK_ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

# Build PTK library
file(GLOB PTK_SOURCES
    ${PROJECT_SOURCE_DIR}/src/sensors/*.cc
    ${PROJECT_SOURCE_DIR}/src/runtime/core/*.cc
    ${PROJECT_SOURCE_DIR}/src/runtime/components/*.cc
    ${PROJECT_SOURCE_DIR}/src/operators/*.cc
)

# Add memory sources only if CUDA is available
if(CUDA_FOUND)
    file(GLOB PTK_MEMORY_SOURCES ${PROJECT_SOURCE_DIR}/src/runtime/memory/*.cc)
    list(APPEND PTK_SOURCES ${PTK_MEMORY_SOURCES})
endif()

add_library(ptk SHARED ${PTK_SOURCES})
ament_target_dependencies(ptk
    rclcpp
    rclcpp_components
)
target_link_libraries(ptk ${OpenCV_LIBS})

# Link CUDA libraries if available
if(CUDA_FOUND)
    target_link_libraries(ptk ${CUDA_LIBRARIES})
endif()

# Register components for composable nodes
rclcpp_components_register_node(ptk
    PLUGIN "ptk::components::Counter"
    EXECUTABLE counter_node
)

rclcpp_components_register_node(ptk
    PLUGIN "ptk::components::Heartbeat"
    EXECUTABLE heartbeat_node
)

rclcpp_components_register_node(ptk
    PLUGIN "ptk::components::FrameDebugger"
    EXECUTABLE frame_debugger_node
)

rclcpp_components_register_node(ptk
    PLUGIN "ptk::components::SyntheticCamera"
    EXECUTABLE synthetic_camera_node
)

rclcpp_components_register_node(ptk
    PLUGIN "ptk::sensors::MacCamera"
    EXECUTABLE mac_camera_node
)

rclcpp_components_register_node(ptk
    PLUGIN "ptk::Preprocessor"
    EXECUTABLE preprocessor_node
)

# Build test apps
add_executable(test_camera src/apps/test_camera.cc)
target_link_libraries(test_camera ptk ${OpenCV_LIBS})
ament_target_dependencies(test_camera rclcpp)

add_executable(test_pipeline src/apps/test_pipeline.cc)
target_link_libraries(test_pipeline ptk ${OpenCV_LIBS})
ament_target_dependencies(test_pipeline rclcpp)

add_executable(test_pipeline_synthetic src/apps/test_pipeline_synthetic.cc)
target_link_libraries(test_pipeline_synthetic ptk ${OpenCV_LIBS})
ament_target_dependencies(test_pipeline_synthetic rclcpp)

# Install
install(TARGETS ptk
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS test_camera test_pipeline test_pipeline_synthetic
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include
)

ament_package()
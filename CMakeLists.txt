cmake_minimum_required(VERSION 3.14)
project(ptk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point OpenCV to Conda installation
set(OpenCV_DIR "$ENV{CONDA_PREFIX}/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Build PTK library
file(GLOB PTK_SOURCES
    ${PROJECT_SOURCE_DIR}/src/sensors/*.cc
    ${PROJECT_SOURCE_DIR}/src/runtime/core/*.cc
    ${PROJECT_SOURCE_DIR}/src/runtime/data/*.cc
)

add_library(ptk STATIC ${PTK_SOURCES})
target_link_libraries(ptk PRIVATE ${OpenCV_LIBS})

# Build test app
add_executable(test_camera src/apps/test_camera.cc)
target_link_libraries(test_camera ptk ${OpenCV_LIBS})
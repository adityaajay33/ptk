# PTK - Perception Toolkit

A high-performance, zero-copy perception pipeline built on ROS 2 composable nodes for edge devices and robotics applications.

## What is PTK?

PTK is a modular perception toolkit that combines:
- **ROS 2 Composable Nodes** - Dynamic loading, introspection, tooling
- **Zero-Copy Architecture** - TensorView for efficient data flow
- **Port-Based Pipeline** - Type-safe component connections
- **Edge-Ready** - Designed for embedded devices (Jetson, RaspberryPi)
- **Inference Support** - ONNX Runtime and TensorRT backends

## Key Features

- **Zero-copy data pipeline** using TensorView/BufferView
- **ROS 2 composable nodes** for all components
- **Thread-safe scheduling** with lifecycle management
- **Image preprocessing operators** (resize, normalize, crop, etc.)
- **Multiple inference backends** (ONNX, TensorRT)
- **Docker support** for reproducible builds
- **Camera abstraction** with real and synthetic sources

## Quick Start

### Prerequisites
- Docker Desktop installed
- 4GB+ RAM allocated to Docker

### Run Test Pipeline

# Clone the repository
git clone https://www.github.com/adityaajay33/ptk
cd ptk

# Build Docker image (first time only)
docker-compose build

# Run synthetic camera test
```bash
docker-compose run --rm -v $(pwd):/output ptk bash -c \
  "cd /ros2_ws && source install/setup.bash && \
   ros2 run ptk test_pipeline_synthetic && \
   cp tensor_output.txt /output/"

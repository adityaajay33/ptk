FROM ros:humble

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -zxvf onnxruntime-linux-x64-1.16.3.tgz && \
    cp onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/ && \
    cp onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-1.16.3.tgz onnxruntime-linux-x64-1.16.3

# Create workspace
WORKDIR /ros2_ws/src/ptk

# Copy source code
COPY . .

# Build the package
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace on container start
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc



# Set default command
CMD ["/bin/bash"]

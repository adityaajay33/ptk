FROM ros:humble

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

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

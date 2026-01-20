#pragma once

#include <rclcpp/rclcpp.hpp>

namespace ptk::components
{

    // Helper to create NodeOptions with intra-process communication enabled
    inline rclcpp::NodeOptions CreateZeroCopyNodeOptions()
    {
        rclcpp::NodeOptions options;
        options.use_intra_process_comms(true);  // Enable intra-process communication
        return options;
    }

} // namespace ptk::components

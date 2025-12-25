// demo.cc
#include <cstdint>
#include <cstddef>
#include <dirent.h>
#include <sys/stat.h>

#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "runtime/core/types.h"
#include "runtime/data/buffer.h"
#include "runtime/data/tensor.h"
#include "operators/cast_uint8_to_float32.h"
#include "operators/rgb_to_gray.h"
#include "operators/cast_float32_to_uint8.h"


// Simple helper to check if a path is a directory.
bool IsDirectory(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISDIR(st.st_mode);
}

// Simple helper to check if a filename has a supported extension.
bool HasSupportedExtension(const std::string& name) {
  std::size_t dot = name.find_last_of('.');
  if (dot == std::string::npos) {
    return false;
  }
  std::string ext = name.substr(dot);
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(c));
  }
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: demo <input_dir> <output_dir>\n"
        << "Example: ./demo ./input_images ./output_gray\n";
    return 1;
  }

  std::string input_dir = argv[1];
  std::string output_dir = argv[2];

  if (!IsDirectory(input_dir)) {
    std::cerr << "Input path is not a directory: " << input_dir << "\n";
    return 1;
  }

  // Create output directory if needed.
  struct stat st;
  if (stat(output_dir.c_str(), &st) != 0) {
    // Try to create directory.
    int res = mkdir(output_dir.c_str(), 0755);
    if (res != 0) {
      std::cerr << "Failed to create output directory: " << output_dir << "\n";
      return 1;
    }
  } else if (!S_ISDIR(st.st_mode)) {
    std::cerr << "Output path exists but is not a directory: "
              << output_dir << "\n";
    return 1;
  }

  DIR* dir = opendir(input_dir.c_str());
  if (dir == nullptr) {
    std::cerr << "Failed to open input directory: " << input_dir << "\n";
    return 1;
  }

  while (true) {
    struct dirent* entry = readdir(dir);
    if (entry == nullptr) {
      break;
    }

    std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }

    if (!HasSupportedExtension(name)) {
      continue;
    }

    std::string in_path = input_dir + "/" + name;
    std::string out_path = output_dir + "/" + name;

    std::cout << "Processing: " << in_path << "\n";

    int width = 0;
    int height = 0;
    int channels_in_file = 0;

    // Force 3 channels (RGB) on load.
    unsigned char* pixels = stbi_load(
        in_path.c_str(), &width, &height, &channels_in_file, 3);
    if (pixels == nullptr) {
      std::cerr << "  Failed to load image\n";
      continue;
    }

    const std::int64_t H = height;
    const std::int64_t W = width;
    const std::int64_t C = 3;

    const std::int64_t num_rgb_elems = H * W * C;
    const std::size_t num_rgb_bytes =
        static_cast<std::size_t>(num_rgb_elems) * sizeof(std::uint8_t);

    // Wrap the loaded uint8 RGB data in a TensorView: [H,W,3] uint8 CPU.
    ptk::data::BufferView rgb_u8_buffer(
        pixels,
        num_rgb_bytes,
        ptk::core::DeviceType::kCpu);

    ptk::data::TensorShape rgb_shape({H, W, C});
    ptk::data::TensorView rgb_u8_tensor(
        rgb_u8_buffer,
        ptk::core::DataType::kUint8,
        rgb_shape);

    // Allocate float32 RGB HWC buffer.
    std::vector<float> rgb_float_storage(
        static_cast<std::size_t>(num_rgb_elems), 0.0f);

    ptk::data::BufferView rgb_float_buffer(
        rgb_float_storage.data(),
        static_cast<std::size_t>(num_rgb_elems) * sizeof(float),
        ptk::core::DeviceType::kCpu);

    ptk::data::TensorView rgb_float_tensor(
        rgb_float_buffer,
        ptk::core::DataType::kFloat32,
        rgb_shape);

    // Cast uint8 -> float32
    {
      ptk::core::Status s =
          ptk::operators::CastUint8ToFloat32(rgb_u8_tensor, &rgb_float_tensor);
      if (!s.ok()) {
        std::cerr << "  CastUint8ToFloat32 failed: "
                  << s.message() << "\n";
        stbi_image_free(pixels);
        continue;
      }
    }

    // Prepare float32 gray tensor [H,W,1].
    const std::int64_t gray_elems = H * W;
    std::vector<float> gray_float_storage(
        static_cast<std::size_t>(gray_elems), 0.0f);

    ptk::data::TensorShape gray_shape({H, W, 1});
    ptk::data::BufferView gray_float_buffer(
        gray_float_storage.data(),
        static_cast<std::size_t>(gray_elems) * sizeof(float),
        ptk::core::DeviceType::kCpu);

    ptk::data::TensorView gray_float_tensor(
        gray_float_buffer,
        ptk::core::DataType::kFloat32,
        gray_shape);

    // RGB float -> Gray float
    {
      ptk::core::Status s =
          ptk::operators::RgbToGray(rgb_float_tensor, &gray_float_tensor);
      if (!s.ok()) {
        std::cerr << "  RgbToGray failed: "
                  << s.message() << "\n";
        stbi_image_free(pixels);
        continue;
      }
    }

    // Cast float32 gray -> uint8 gray for saving.
    std::vector<std::uint8_t> gray_u8_storage(
        static_cast<std::size_t>(gray_elems), 0);

    ptk::data::BufferView gray_u8_buffer(
        gray_u8_storage.data(),
        static_cast<std::size_t>(gray_elems) * sizeof(std::uint8_t),
        ptk::core::DeviceType::kCpu);

    ptk::data::TensorView gray_u8_tensor(
        gray_u8_buffer,
        ptk::core::DataType::kUint8,
        gray_shape);

    {
      ptk::core::Status s =
          ptk::operators::CastFloat32ToUint8(gray_float_tensor, &gray_u8_tensor);
      if (!s.ok()) {
        std::cerr << "  CastFloat32ToUint8 failed: "
                  << s.message() << "\n";
        stbi_image_free(pixels);
        continue;
      }
    }

    // Write grayscale image to output folder.
    int stride_in_bytes = width;  // for 1 channel uint8
    int write_ok = stbi_write_png(
        out_path.c_str(),
        width,
        height,
        1,  // channels
        gray_u8_storage.data(),
        stride_in_bytes);

    if (!write_ok) {
      std::cerr << "  Failed to write output image: " << out_path << "\n";
    } else {
      std::cout << "  Wrote: " << out_path << "\n";
    }

    stbi_image_free(pixels);
  }

  closedir(dir);
  return 0;
}
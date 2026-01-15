#include "engines/engine_tensor.h"

#include <sstream>
#include <algorithm>
#include <numeric>

namespace ptk::engine
{

  // EngineTensor Implementation

  EngineTensor::EngineTensor(const data::TensorView &tensor_view)
      : tensor_view_(tensor_view)
  {
    metadata_.name = "unnamed";
    metadata_.is_input = false;
    metadata_.is_output = false;
    metadata_.is_dynamic_shape = false;
  }

  EngineTensor::EngineTensor(const data::TensorView &tensor_view,
                             const TensorMetadata &metadata)
      : tensor_view_(tensor_view), metadata_(metadata) {}

  core::Status EngineTensor::ValidateShape(
      const data::TensorShape &expected_shape) const
  {
    if (tensor_view_.shape().rank() != expected_shape.rank())
    {
      return core::Status(core::StatusCode::kInvalidArgument,
                          "EngineTensor: Shape rank mismatch");
    }

    for (size_t i = 0; i < expected_shape.rank(); ++i)
    {
      if (tensor_view_.shape().dim(i) != expected_shape.dim(i))
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "EngineTensor: Shape dimension mismatch");
      }
    }

    return core::Status::Ok();
  }

  core::Status EngineTensor::ValidateShape(
      const std::vector<int64_t> &expected_dims) const
  {
    data::TensorShape expected_shape(expected_dims);
    return ValidateShape(expected_shape);
  }

  core::Status EngineTensor::SetDynamicShapeBounds(
      const std::vector<size_t> &min_dims,
      const std::vector<size_t> &max_dims)
  {
    if (min_dims.size() != max_dims.size())
    {
      return core::Status(core::StatusCode::kInvalidArgument,
                          "EngineTensor: Min and max dims must have same size");
    }

    metadata_.is_dynamic_shape = true;
    metadata_.min_dims = min_dims;
    metadata_.max_dims = max_dims;

    return core::Status::Ok();
  }

  core::Status EngineTensor::ValidateDynamicShape() const
  {
    if (!metadata_.is_dynamic_shape)
    {
      return core::Status::Ok();
    }

    const auto &shape_dims = tensor_view_.shape().dims();

    if (shape_dims.size() != metadata_.min_dims.size())
    {
      return core::Status(core::StatusCode::kInvalidArgument,
                          "EngineTensor: Dynamic shape rank mismatch");
    }

    for (size_t i = 0; i < shape_dims.size(); ++i)
    {
      size_t dim = shape_dims[i];
      if (dim < metadata_.min_dims[i] || dim > metadata_.max_dims[i])
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "EngineTensor: Dimension out of dynamic bounds");
      }
    }

    return core::Status::Ok();
  }

  core::Status EngineTensor::ValidateDataType(core::DataType expected_dtype) const
  {
    if (tensor_view_.dtype() != expected_dtype)
    {
      return core::Status(core::StatusCode::kInvalidArgument,
                          "EngineTensor: Data type mismatch");
    }
    return core::Status::Ok();
  }

  core::Status EngineTensor::ValidateDevice(
      core::DeviceType expected_device) const
  {
    if (tensor_view_.buffer().device_type() != expected_device)
    {
      return core::Status(core::StatusCode::kInvalidArgument,
                          "EngineTensor: Device mismatch");
    }
    return core::Status::Ok();
  }

  core::Status EngineTensor::Validate() const
  {
    auto status = ValidateDataType(tensor_view_.dtype());
    if (!status.ok())
      return status;

    status = ValidateDevice(tensor_view_.buffer().device_type());
    if (!status.ok())
      return status;

    if (metadata_.is_dynamic_shape)
    {
      status = ValidateDynamicShape();
      if (!status.ok())
        return status;
    }

    return core::Status::Ok();
  }

  std::string EngineTensor::ToString() const
  {
    std::ostringstream oss;
    oss << "EngineTensor {\n";
    oss << "  name: " << metadata_.name << "\n";
    oss << "  shape: [";

    const auto &dims = tensor_view_.shape().dims();
    for (size_t i = 0; i < dims.size(); ++i)
    {
      oss << dims[i];
      if (i < dims.size() - 1)
        oss << ", ";
    }
    oss << "]\n";

    oss << "  dtype: " << static_cast<int>(tensor_view_.dtype()) << "\n";
    oss << "  device: " << static_cast<int>(tensor_view_.buffer().device_type())
        << "\n";
    oss << "  is_input: " << (metadata_.is_input ? "true" : "false") << "\n";
    oss << "  is_output: " << (metadata_.is_output ? "true" : "false") << "\n";
    oss << "  is_dynamic: "
        << (metadata_.is_dynamic_shape ? "true" : "false") << "\n";
    oss << "  bytes: " << tensor_view_.buffer().size_bytes() << "\n";
    oss << "}\n";

    return oss.str();
  }

  // Shape Inference Implementation

  namespace shape_inference
  {

    data::TensorShape ShapeCalculator::ElementWise(
        const data::TensorShape &input)
    {
      return input;
    }

    data::TensorShape ShapeCalculator::Reshape(
        const data::TensorShape &input,
        const std::vector<int64_t> &new_shape)
    {
      return data::TensorShape(new_shape);
    }

    data::TensorShape ShapeCalculator::Transpose(
        const data::TensorShape &input,
        const std::vector<int> &perm)
    {
      std::vector<int64_t> output_dims(perm.size());

      for (size_t i = 0; i < perm.size(); ++i)
      {
        output_dims[i] = input.dim(perm[i]);
      }

      return data::TensorShape(output_dims);
    }

    data::TensorShape ShapeCalculator::Reduce(const data::TensorShape &input,
                                              const std::vector<int> &axes,
                                              bool keep_dims)
    {
      auto output_dims = input.dims();

      if (keep_dims)
      {
        for (int axis : axes)
        {
          if (axis >= 0 && axis < static_cast<int>(output_dims.size()))
          {
            output_dims[axis] = 1;
          }
        }
      }
      else
      {
        std::vector<bool> to_remove(output_dims.size(), false);
        for (int axis : axes)
        {
          if (axis >= 0 && axis < static_cast<int>(output_dims.size()))
          {
            to_remove[axis] = true;
          }
        }

        std::vector<int64_t> result;
        for (size_t i = 0; i < output_dims.size(); ++i)
        {
          if (!to_remove[i])
          {
            result.push_back(output_dims[i]);
          }
        }
        output_dims = result;
      }

      return data::TensorShape(output_dims);
    }

    core::Status ShapeCalculator::Concatenate(
        const std::vector<data::TensorShape> &inputs,
        int axis,
        data::TensorShape *output)
    {
      if (inputs.empty())
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: Empty input list");
      }

      if (!output)
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: Output is null");
      }

      auto output_dims = inputs[0].dims();

      for (size_t i = 1; i < inputs.size(); ++i)
      {
        if (inputs[i].rank() != output_dims.size())
        {
          return core::Status(core::StatusCode::kInvalidArgument,
                              "ShapeCalculator: Rank mismatch in concatenate");
        }

        output_dims[axis] += inputs[i].dim(axis);
      }

      *output = data::TensorShape(output_dims);
      return core::Status::Ok();
    }

    core::Status ShapeCalculator::MatMul(const data::TensorShape &lhs,
                                         const data::TensorShape &rhs,
                                         data::TensorShape *output)
    {
      if (!output)
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: Output is null");
      }

      if (lhs.rank() < 2 || rhs.rank() < 2)
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: MatMul requires rank >= 2");
      }

      if (lhs.dim(lhs.rank() - 1) != rhs.dim(rhs.rank() - 2))
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: MatMul dimension mismatch");
      }

      std::vector<int64_t> output_dims;
      for (size_t i = 0; i < lhs.rank() - 1; ++i)
      {
        output_dims.push_back(lhs.dim(i));
      }
      for (size_t i = rhs.rank() - 1; i < rhs.rank(); ++i)
      {
        output_dims.push_back(rhs.dim(i));
      }

      *output = data::TensorShape(output_dims);
      return core::Status::Ok();
    }

    core::Status ShapeCalculator::BroadcastBinary(
        const data::TensorShape &lhs,
        const data::TensorShape &rhs,
        data::TensorShape *output)
    {
      if (!output)
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: Output is null");
      }

      auto lhs_dims = lhs.dims();
      auto rhs_dims = rhs.dims();

      size_t max_rank = std::max(lhs_dims.size(), rhs_dims.size());

      // Pad with 1s
      while (lhs_dims.size() < max_rank)
      {
        lhs_dims.insert(lhs_dims.begin(), 1);
      }
      while (rhs_dims.size() < max_rank)
      {
        rhs_dims.insert(rhs_dims.begin(), 1);
      }

      std::vector<int64_t> result_dims;
      for (size_t i = 0; i < max_rank; ++i)
      {
        int64_t lhs_dim = lhs_dims[i];
        int64_t rhs_dim = rhs_dims[i];

        if (lhs_dim == 1)
        {
          result_dims.push_back(rhs_dim);
        }
        else if (rhs_dim == 1)
        {
          result_dims.push_back(lhs_dim);
        }
        else if (lhs_dim == rhs_dim)
        {
          result_dims.push_back(lhs_dim);
        }
        else
        {
          return core::Status(core::StatusCode::kInvalidArgument,
                              "ShapeCalculator: Cannot broadcast shapes");
        }
      }

      *output = data::TensorShape(result_dims);
      return core::Status::Ok();
    }

    data::TensorShape ShapeCalculator::Flatten(const data::TensorShape &input,
                                               int start_axis,
                                               int end_axis)
    {
      auto dims = input.dims();

      if (end_axis == -1)
      {
        end_axis = dims.size() - 1;
      }

      int64_t flattened_size = 1;
      for (int i = start_axis; i <= end_axis; ++i)
      {
        flattened_size *= dims[i];
      }

      std::vector<int64_t> output_dims;
      for (int i = 0; i < start_axis; ++i)
      {
        output_dims.push_back(dims[i]);
      }
      output_dims.push_back(flattened_size);
      for (size_t i = end_axis + 1; i < dims.size(); ++i)
      {
        output_dims.push_back(dims[i]);
      }

      return data::TensorShape(output_dims);
    }

    core::Status ShapeCalculator::Squeeze(const data::TensorShape &input,
                                          const std::vector<int> &axes,
                                          data::TensorShape *output)
    {
      if (!output)
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: Output is null");
      }

      auto dims = input.dims();
      std::vector<bool> to_squeeze(dims.size(), false);

      for (int axis : axes)
      {
        if (axis >= 0 && axis < static_cast<int>(dims.size()))
        {
          if (dims[axis] != 1)
          {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "ShapeCalculator: Cannot squeeze dimension != 1");
          }
          to_squeeze[axis] = true;
        }
      }

      std::vector<int64_t> result_dims;
      for (size_t i = 0; i < dims.size(); ++i)
      {
        if (!to_squeeze[i])
        {
          result_dims.push_back(dims[i]);
        }
      }

      *output = data::TensorShape(result_dims);
      return core::Status::Ok();
    }

    core::Status ShapeCalculator::Unsqueeze(const data::TensorShape &input,
                                            const std::vector<int> &axes,
                                            data::TensorShape *output)
    {
      if (!output)
      {
        return core::Status(core::StatusCode::kInvalidArgument,
                            "ShapeCalculator: Output is null");
      }

      auto result_dims = input.dims();

      std::vector<int> sorted_axes = axes;
      std::sort(sorted_axes.rbegin(), sorted_axes.rend());

      for (int axis : sorted_axes)
      {
        result_dims.insert(result_dims.begin() + axis, 1);
      }

      *output = data::TensorShape(result_dims);
      return core::Status::Ok();
    }

    data::TensorShape ShapeCalculator::Slice(const data::TensorShape &input,
                                             const std::vector<int> &starts,
                                             const std::vector<int> &ends)
    {
      auto dims = input.dims();
      std::vector<int64_t> result_dims;

      for (size_t i = 0; i < starts.size() && i < dims.size(); ++i)
      {
        int start = starts[i];
        int end = ends[i];

        if (start < 0)
          start += dims[i];
        if (end < 0)
          end += dims[i];

        result_dims.push_back(end - start);
      }

      return data::TensorShape(result_dims);
    }

    core::Status ShapeCalculator::BroadcastShapes(
        const data::TensorShape &shape1,
        const data::TensorShape &shape2,
        data::TensorShape *output)
    {
      return BroadcastBinary(shape1, shape2, output);
    }

  } // namespace shape_inference

  // EngineTensorBatch Implementation

  core::Status EngineTensorBatch::ValidateAll() const
  {
    for (const auto &tensor : tensors_)
    {
      auto status = tensor.Validate();
      if (!status.ok())
      {
        return status;
      }
    }
    return core::Status::Ok();
  }

  std::vector<std::string> EngineTensorBatch::GetNames() const
  {
    std::vector<std::string> names;
    for (const auto &tensor : tensors_)
    {
      names.push_back(tensor.GetName());
    }
    return names;
  }

  std::vector<core::DataType> EngineTensorBatch::GetDataTypes() const
  {
    std::vector<core::DataType> types;
    for (const auto &tensor : tensors_)
    {
      types.push_back(tensor.GetDataType());
    }
    return types;
  }

} // namespace ptk::engine

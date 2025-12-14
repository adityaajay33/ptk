#include "runtime/memory/device_transfer.h"

#include <cstring>
#include <iostream>

#ifndef __APPLE__
#include <cuda_runtime.h>
#endif

namespace ptk::memory
{
    // PinnedMemoryPool Implementation

    PinnedMemoryPool::~PinnedMemoryPool() { Clear(); }

    core::Status PinnedMemoryPool::Allocate(size_t bytes, void **ptr)
    {
        if (!ptr)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "PinnedMemoryPool: Output pointer is null");
        }

        if (bytes == 0)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "PinnedMemoryPool: Cannot allocate zero bytes");
        }

#ifndef __APPLE__
        cudaError_t cuda_status = cudaMallocHost(ptr, bytes);
        if (cuda_status != cudaSuccess)
        {
            return core::Status(
                core::StatusCode::kInternal,
                std::string("PinnedMemoryPool: cudaMallocHost failed: ") +
                    cudaGetErrorString(cuda_status));
        }
#else
        // Fallback to aligned malloc on macOS
        if (posix_memalign(ptr, 256, bytes) != 0)
        {
            return core::Status(core::StatusCode::kInternal,
                                "PinnedMemoryPool: posix_memalign failed");
        }
#endif

        allocations_.emplace_back(*ptr, bytes);
        total_allocated_ += bytes;

        return core::Status::Ok();
    }

    core::Status PinnedMemoryPool::Free(void *ptr)
    {
        if (!ptr)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "PinnedMemoryPool: Pointer is null");
        }

        auto it = std::find_if(
            allocations_.begin(), allocations_.end(),
            [ptr](const auto &pair)
            { return pair.first == ptr; });

        if (it == allocations_.end())
        {
            return core::Status(core::StatusCode::kFailedPrecondition,
                                "PinnedMemoryPool: Pointer not found in allocations");
        }

        size_t bytes = it->second;

#ifndef __APPLE__
        cudaError_t cuda_status = cudaFreeHost(ptr);
        if (cuda_status != cudaSuccess)
        {
            return core::Status(
                core::StatusCode::kInternal,
                std::string("PinnedMemoryPool: cudaFreeHost failed: ") +
                    cudaGetErrorString(cuda_status));
        }
#else
        std::free(ptr);
#endif

        allocations_.erase(it);
        total_allocated_ -= bytes;

        return core::Status::Ok();
    }

    void PinnedMemoryPool::Clear()
    {
        for (auto &allocation : allocations_)
        {
#ifndef __APPLE__
            cudaFreeHost(allocation.first);
#else
            std::free(allocation.first);
#endif
        }
        allocations_.clear();
        total_allocated_ = 0;
    }

    // DeviceAwareTensorAllocator Implementation

    DeviceAwareTensorAllocator::DeviceAwareTensorAllocator(
        PinnedMemoryPool *pinned_pool)
        : pinned_pool_(pinned_pool) {}

    size_t DeviceAwareTensorAllocator::GetElementSize(core::DataType dtype) const
    {
        switch (dtype)
        {
        case core::DataType::kUint8:
            return 1;
        case core::DataType::kInt32:
        case core::DataType::kFloat32:
            return 4;
        case core::DataType::kInt64:
        case core::DataType::kFloat64:
            return 8;
        default:
            return 0;
        }
    }

    core::Status DeviceAwareTensorAllocator::AllocateTensor(
        const data::TensorShape &shape,
        core::DataType dtype,
        core::DeviceType device,
        data::TensorView *output)
    {
        if (!output)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceAwareTensorAllocator: Output is null");
        }

        size_t element_size = GetElementSize(dtype);
        if (element_size == 0)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceAwareTensorAllocator: Unsupported data type");
        }

        size_t total_elements = 1;
        for (const auto &dim : shape.dims())
        {
            total_elements *= dim;
        }

        size_t bytes = total_elements * element_size;

        void *ptr = nullptr;

        if (device == core::DeviceType::kCpu)
        {
            // CPU: aligned malloc
            if (posix_memalign(&ptr, 64, bytes) != 0)
            {
                return core::Status(core::StatusCode::kInternal,
                                    "DeviceAwareTensorAllocator: CPU allocation failed");
            }
        }
        else if (device == core::DeviceType::kCuda)
        {
            // GPU: use pinned memory pool if available
            if (pinned_pool_)
            {
                core::Status status = pinned_pool_->Allocate(bytes, &ptr);
                if (!status.ok())
                {
                    return status;
                }
            }
            else
            {
                // Fallback to regular malloc
                ptr = std::malloc(bytes);
                if (!ptr)
                {
                    return core::Status(core::StatusCode::kInternal,
                                        "DeviceAwareTensorAllocator: GPU allocation failed");
                }
            }
        }
        else
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceAwareTensorAllocator: Unknown device type");
        }

        data::BufferView buffer(ptr, bytes, device);
        *output = data::TensorView(buffer, dtype, shape);

        return core::Status::Ok();
    }

    core::Status DeviceAwareTensorAllocator::AllocateTensorOwned(
        const data::TensorShape &shape,
        core::DataType dtype,
        core::DeviceType device,
        std::unique_ptr<data::TensorView> *output)
    {
        if (!output)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceAwareTensorAllocator: Output is null");
        }

        data::TensorView tensor;
        core::Status status =
            AllocateTensor(shape, dtype, device, &tensor);
        if (!status.ok())
        {
            return status;
        }

        *output = std::make_unique<data::TensorView>(tensor);
        return core::Status::Ok();
    }

    core::Status DeviceTransfer::ValidateDevices(
        const data::TensorView &tensor,
        core::DeviceType expected_device)
    {
        if (tensor.buffer().device_type() != expected_device)
        {
            return core::Status(
                core::StatusCode::kInvalidArgument,
                "DeviceTransfer: Device mismatch (expected vs actual)");
        }
        return core::Status::Ok();
    }

    core::Status DeviceTransfer::ValidateSizes(const data::TensorView &src,
                                               const data::TensorView &dst)
    {
        if (src.buffer().size_bytes() != dst.buffer().size_bytes())
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Size mismatch between tensors");
        }
        return core::Status::Ok();
    }

    core::Status DeviceTransfer::ValidateDtypes(const data::TensorView &src,
                                                const data::TensorView &dst)
    {
        if (src.dtype() != dst.dtype())
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Data type mismatch between tensors");
        }
        return core::Status::Ok();
    }

    core::Status DeviceTransfer::CpuToGpu(const data::TensorView &src,
                                          data::TensorView *dst)
    {
        if (!dst)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Destination is null");
        }

        // Validate source is CPU
        core::Status status = ValidateDevices(src, core::DeviceType::kCpu);
        if (!status.ok())
        {
            return status;
        }

        // Validate destination is GPU
        status = ValidateDevices(*dst, core::DeviceType::kCuda);
        if (!status.ok())
        {
            return status;
        }

        // Validate sizes and types match
        status = ValidateSizes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDtypes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

#ifndef __APPLE__
        cudaError_t cuda_status = cudaMemcpy(
            dst->buffer().data(),
            const_cast<void *>(src.buffer().data()),
            src.buffer().size_bytes(),
            cudaMemcpyHostToDevice);

        if (cuda_status != cudaSuccess)
        {
            return core::Status(
                core::StatusCode::kInternal,
                std::string("DeviceTransfer: CPU->GPU copy failed: ") +
                    cudaGetErrorString(cuda_status));
        }
#else
        std::memcpy(dst->buffer().data(),
                    const_cast<void *>(src.buffer().data()),
                    src.buffer().size_bytes());
#endif

        return core::Status::Ok();
    }

    core::Status DeviceTransfer::GpuToCpu(const data::TensorView &src,
                                          data::TensorView *dst)
    {
        if (!dst)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Destination is null");
        }

        // Validate source is GPU
        core::Status status = ValidateDevices(src, core::DeviceType::kCuda);
        if (!status.ok())
        {
            return status;
        }

        // Validate destination is CPU
        status = ValidateDevices(*dst, core::DeviceType::kCpu);
        if (!status.ok())
        {
            return status;
        }

        // Validate sizes and types match
        status = ValidateSizes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDtypes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

#ifndef __APPLE__
        cudaError_t cuda_status = cudaMemcpy(
            dst->buffer().data(),
            const_cast<void *>(src.buffer().data()),
            src.buffer().size_bytes(),
            cudaMemcpyDeviceToHost);

        if (cuda_status != cudaSuccess)
        {
            return core::Status(
                core::StatusCode::kInternal,
                std::string("DeviceTransfer: GPU->CPU copy failed: ") +
                    cudaGetErrorString(cuda_status));
        }
#else
        std::memcpy(dst->buffer().data(),
                    const_cast<void *>(src.buffer().data()),
                    src.buffer().size_bytes());
#endif

        return core::Status::Ok();
    }

    core::Status DeviceTransfer::CpuToGpuAsync(const data::TensorView &src,
                                               data::TensorView *dst,
                                               void *cuda_stream)
    {
        if (!dst)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Destination is null");
        }

        // Validate source is CPU
        core::Status status = ValidateDevices(src, core::DeviceType::kCpu);
        if (!status.ok())
        {
            return status;
        }

        // Validate destination is GPU
        status = ValidateDevices(*dst, core::DeviceType::kCuda);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateSizes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDtypes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

#ifndef __APPLE__
        cudaStream_t stream = cuda_stream ? *static_cast<cudaStream_t *>(cuda_stream)
                                          : cudaStreamDefault;

        cudaError_t cuda_status = cudaMemcpyAsync(
            dst->buffer().data(),
            const_cast<void *>(src.buffer().data()),
            src.buffer().size_bytes(),
            cudaMemcpyHostToDevice,
            stream);

        if (cuda_status != cudaSuccess)
        {
            return core::Status(
                core::StatusCode::kInternal,
                std::string("DeviceTransfer: Async CPU->GPU copy failed: ") +
                    cudaGetErrorString(cuda_status));
        }
#else
        // macOS: fall back to sync copy
        std::memcpy(dst->buffer().data(),
                    const_cast<void *>(src.buffer().data()),
                    src.buffer().size_bytes());
#endif

        return core::Status::Ok();
    }

    core::Status DeviceTransfer::GpuToCpuAsync(const data::TensorView &src,
                                               data::TensorView *dst,
                                               void *cuda_stream)
    {
        if (!dst)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Destination is null");
        }

        core::Status status = ValidateDevices(src, core::DeviceType::kCuda);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDevices(*dst, core::DeviceType::kCpu);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateSizes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDtypes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

#ifndef __APPLE__
        cudaStream_t stream = cuda_stream ? *static_cast<cudaStream_t *>(cuda_stream)
                                          : cudaStreamDefault;

        cudaError_t cuda_status = cudaMemcpyAsync(
            dst->buffer().data(),
            const_cast<void *>(src.buffer().data()),
            src.buffer().size_bytes(),
            cudaMemcpyDeviceToHost,
            stream);

        if (cuda_status != cudaSuccess)
        {
            return core::Status(
                core::StatusCode::kInternal,
                std::string("DeviceTransfer: Async GPU->CPU copy failed: ") +
                    cudaGetErrorString(cuda_status));
        }
#else
        // macOS: fall back to sync copy
        std::memcpy(dst->buffer().data(),
                    const_cast<void *>(src.buffer().data()),
                    src.buffer().size_bytes());
#endif

        return core::Status::Ok();
    }

    core::Status DeviceTransfer::Copy(const data::TensorView &src,
                                      data::TensorView *dst)
    {
        if (!dst)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Destination is null");
        }

        core::DeviceType src_device = src.buffer().device_type();
        core::DeviceType dst_device = dst->buffer().device_type();

        // Validate sizes and types
        core::Status status = ValidateSizes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDtypes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        // Handle all cases
        if (src_device == core::DeviceType::kCpu &&
            dst_device == core::DeviceType::kCuda)
        {
            return CpuToGpu(src, dst);
        }
        else if (src_device == core::DeviceType::kCuda &&
                 dst_device == core::DeviceType::kCpu)
        {
            return GpuToCpu(src, dst);
        }
        else if (src_device == core::DeviceType::kCpu &&
                 dst_device == core::DeviceType::kCpu)
        {
            // CPU to CPU
            std::memcpy(dst->buffer().data(),
                        const_cast<void *>(src.buffer().data()),
                        src.buffer().size_bytes());
            return core::Status::Ok();
        }
        else
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Unsupported device combination");
        }
    }

    core::Status DeviceTransfer::CopyAsync(const data::TensorView &src,
                                           data::TensorView *dst,
                                           void *cuda_stream)
    {
        if (!dst)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Destination is null");
        }

        core::DeviceType src_device = src.buffer().device_type();
        core::DeviceType dst_device = dst->buffer().device_type();

        core::Status status = ValidateSizes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        status = ValidateDtypes(src, *dst);
        if (!status.ok())
        {
            return status;
        }

        if (src_device == core::DeviceType::kCpu &&
            dst_device == core::DeviceType::kCuda)
        {
            return CpuToGpuAsync(src, dst, cuda_stream);
        }
        else if (src_device == core::DeviceType::kCuda &&
                 dst_device == core::DeviceType::kCpu)
        {
            return GpuToCpuAsync(src, dst, cuda_stream);
        }
        else if (src_device == core::DeviceType::kCpu &&
                 dst_device == core::DeviceType::kCpu)
        {
            // CPU to CPU (no async needed)
            std::memcpy(dst->buffer().data(),
                        const_cast<void *>(src.buffer().data()),
                        src.buffer().size_bytes());
            return core::Status::Ok();
        }
        else
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "DeviceTransfer: Unsupported device combination");
        }
    }

} // namespace ptk::memory
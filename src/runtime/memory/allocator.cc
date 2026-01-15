#include "runtime/memory/allocator.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#ifndef __APPLE__
#include <cuda_runtime.h>
#endif

namespace ptk::memory
{

    // CPU Allocator Implementation

    size_t CpuAllocator::AlignSize(size_t size) const
    {
        const size_t alignment = GetAlignmentForDevice(core::DeviceType::kCpu);
        return (size + alignment - 1) & ~(alignment - 1); // Round up to alignment
    }

    core::Status CpuAllocator::Allocate(size_t num_bytes, core::DeviceType device,
                                        void **out_ptr)
    {
        if (device != core::DeviceType::kCpu)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "CpuAllocator: Device mismatch");
        }

        if (num_bytes == 0)
        {
            *out_ptr = nullptr;
            return core::Status::Ok();
        }

        const size_t aligned_bytes = AlignSize(num_bytes);

        // Use aligned allocation for better performance
        void *ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(aligned_bytes, GetAlignmentForDevice(device));
#else
        const int alignment_code =
            posix_memalign(&ptr, GetAlignmentForDevice(device), aligned_bytes);
        if (alignment_code != 0)
        {
            ptr = nullptr;
        }
#endif

        if (!ptr)
        {
            return core::Status(core::StatusCode::kInternal,
                                "CpuAllocator: malloc failed");
        }

        *out_ptr = ptr;
        return core::Status::Ok();
    }

    core::Status CpuAllocator::Deallocate(void *ptr, core::DeviceType device)
    {
        if (device != core::DeviceType::kCpu)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "CpuAllocator: Device mismatch");
        }

        if (ptr)
        {
#ifdef _MSC_VER
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }

        return core::Status::Ok();
    }

    // CUDA Allocator Implementation

#ifndef __APPLE__

    void CudaDeleter::operator()(void *ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    CudaAllocator::CudaAllocator(int device_id) : device_id_(device_id) {}

    CudaAllocator::~CudaAllocator() = default;

    size_t CudaAllocator::AlignSize(size_t size) const
    {
        const size_t alignment = GetAlignmentForDevice(core::DeviceType::kGpu);
        return (size + alignment - 1) & ~(alignment - 1);
    }

    core::Status CudaAllocator::Allocate(size_t num_bytes, core::DeviceType device,
                                         void **out_ptr)
    {
        if (device != core::DeviceType::kCuda)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "CudaAllocator: Device mismatch");
        }

        if (num_bytes == 0)
        {
            *out_ptr = nullptr;
            return core::Status::Ok();
        }

        const size_t aligned_bytes = AlignSize(num_bytes);

        void *ptr = nullptr;
        cudaError_t cuda_status = cudaMalloc(&ptr, aligned_bytes);

        if (cuda_status != cudaSuccess)
        {
            std::cerr << "CudaAllocator::Allocate - cudaMalloc failed: "
                      << cudaGetErrorString(cuda_status) << "\n";
            return core::Status(core::StatusCode::kInternal,
                                "CudaAllocator: cudaMalloc failed");
        }

        *out_ptr = ptr;
        return core::Status::Ok();
    }

    core::Status CudaAllocator::Deallocate(void *ptr, core::DeviceType device)
    {
        if (device != core::DeviceType::kCuda)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "CudaAllocator: Device mismatch");
        }

        if (ptr)
        {
            cudaError_t cuda_status = cudaFree(ptr);
            if (cuda_status != cudaSuccess)
            {
                std::cerr << "CudaAllocator::Deallocate - cudaFree failed: "
                          << cudaGetErrorString(cuda_status) << "\n";
                return core::Status(core::StatusCode::kInternal,
                                    "CudaAllocator: cudaFree failed");
            }
        }

        return core::Status::Ok();
    }

#endif

    // Memory Pool Implementation

    MemoryPool::MemoryPool(Allocator *allocator) : allocator_(allocator) {}

    MemoryPool::~MemoryPool()
    {
        Clear();
    }

    core::Status MemoryPool::Allocate(size_t num_bytes, void **out_ptr)
    {
        if (!allocator_)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "MemoryPool: Allocator is null");
        }

        // Try to find a reusable buffer
        auto it = pool_.find(num_bytes);
        if (it != pool_.end() && !it->second.empty())
        {
            PooledBuffer buffer = it->second.back();
            it->second.pop_back();
            *out_ptr = buffer.ptr;
            allocated_[buffer.ptr] = num_bytes;
            return core::Status::Ok();
        }

        // No reusable buffer, allocate new
        void *ptr = nullptr;
        const auto status =
            allocator_->Allocate(num_bytes, allocator_->GetDeviceType(), &ptr);

        if (!status.ok())
        {
            return status;
        }

        allocated_[ptr] = num_bytes;
        *out_ptr = ptr;
        return core::Status::Ok();
    }

    core::Status MemoryPool::Release(void *ptr, size_t num_bytes)
    {
        if (!ptr)
        {
            return core::Status::Ok();
        }

        auto it = allocated_.find(ptr);
        if (it == allocated_.end())
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "MemoryPool: Unknown buffer");
        }

        allocated_.erase(it);
        pool_[num_bytes].push_back({ptr, num_bytes});
        return core::Status::Ok();
    }

    size_t MemoryPool::GetAllocatedBytes() const
    {
        size_t total = 0;
        for (const auto &pair : allocated_)
        {
            total += pair.second;
        }
        return total;
    }

    size_t MemoryPool::GetReservedBytes() const
    {
        size_t total = 0;
        for (const auto &size_pool_pair : pool_)
        {
            for (const auto &buffer : size_pool_pair.second)
            {
                total += buffer.size;
            }
        }
        return total;
    }

    void MemoryPool::Clear()
    {
        for (auto &size_pool_pair : pool_)
        {
            for (auto &buffer : size_pool_pair.second)
            {
                allocator_->Deallocate(buffer.ptr, allocator_->GetDeviceType());
            }
        }
        pool_.clear();
        allocated_.clear();
    }

    // Unified Tensor Allocation API

    core::Status AllocateTensor(const data::TensorShape &shape, core::DataType dtype,
                                core::DeviceType device, Allocator *allocator,
                                data::TensorView *out_tensor)
    {
        if (!allocator)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "AllocateTensor: Allocator is null");
        }

        const size_t total_bytes = CalculateTensorBytes(shape, dtype);

        void *ptr = nullptr;
        const auto status = allocator->Allocate(total_bytes, device, &ptr);
        if (!status.ok())
        {
            return status;
        }

        if (!ptr && total_bytes > 0)
        {
            return core::Status(core::StatusCode::kInternal,
                                "AllocateTensor: Allocation returned null pointer");
        }

        data::BufferView buffer_view(ptr, total_bytes, device);
        *out_tensor = data::TensorView(buffer_view, dtype, shape);

        return core::Status::Ok();
    }

    core::Status AllocateTensorOwned(const data::TensorShape &shape,
                                     core::DataType dtype,
                                     core::DeviceType device, Allocator *allocator,
                                     std::unique_ptr<data::TensorView> *out_tensor)
    {
        if (!allocator)
        {
            return core::Status(core::StatusCode::kInvalidArgument,
                                "AllocateTensorOwned: Allocator is null");
        }

        data::TensorView tensor;
        const auto status =
            AllocateTensor(shape, dtype, device, allocator, &tensor);
        if (!status.ok())
        {
            return status;
        }

        *out_tensor = std::make_unique<data::TensorView>(tensor);

        return core::Status::Ok();
    }

} // namespace ptk::memory

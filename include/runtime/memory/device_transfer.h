#pragma once

#include <memory>
#include <vector>

#include "runtime/core/status.h"
#include "runtime/core/types.h"
#include "runtime/data/tensor.h"

//primarily used for cuda operations - save for now

namespace ptk::memory
{

    class PinnedMemoryPool
    {
    public:
        PinnedMemoryPool() = default;
        ~PinnedMemoryPool();

        core::Status Allocate(size_t bytes, void **ptr);
        core::Status Free(void *ptr);

        size_t GetTotalAllocated() const { return total_allocated_; };

        void Clear();

    private:
        std::vector<std::pair<void *, size_t>> allocations_;
        size_t total_allocated_ = 0;
    };

    class DeviceAwareTensorAllocator
    {
    public:
        explicit DeviceAwareTensorAllocator(PinnedMemoryPool *pinned_pool = nullptr);
        ~DeviceAwareTensorAllocator() = default;

        core::Status AllocateTensor(const data::TensorShape &shape,
                                    core::DataType dtype,
                                    core::DeviceType device,
                                    data::TensorView *output);

        core::Status AllocateTensorOwned(
            const data::TensorShape &shape,
            core::DataType dtype,
            core::DeviceType device,
            std::unique_ptr<data::TensorView> *output);

    private:
        PinnedMemoryPool *pinned_pool_;
        size_t GetElementSize(core::DataType dtype) const;
    };

    // Device transfer operations
    class DeviceTransfer
    {
    public:
        DeviceTransfer() = default;
        ~DeviceTransfer() = default;

        // Copy from CPU to GPU (synchronous)
        static core::Status CpuToGpu(const data::TensorView &src,
                                     data::TensorView *dst);

        // Copy from GPU to CPU (synchronous)
        static core::Status GpuToCpu(const data::TensorView &src,
                                     data::TensorView *dst);

        // Async copy from CPU to GPU with CUDA stream
        static core::Status CpuToGpuAsync(const data::TensorView &src,
                                          data::TensorView *dst,
                                          void *cuda_stream);

        // Async copy from GPU to CPU with CUDA stream
        static core::Status GpuToCpuAsync(const data::TensorView &src,
                                          data::TensorView *dst,
                                          void *cuda_stream);

        // Validate tensor devices match expected types
        static core::Status ValidateDevices(const data::TensorView &tensor,
                                            core::DeviceType expected_device);

        // Validate tensor sizes match
        static core::Status ValidateSizes(const data::TensorView &src, const data::TensorView &dst);

        // Validate tensor dtypes match
        static core::Status ValidateDtypes(const data::TensorView &src,
                                           const data::TensorView &dst);

        // Copy between any devices (handles all cases)
        static core::Status Copy(const data::TensorView &src,
                                 data::TensorView *dst);

        // Copy async between any devices
        static core::Status CopyAsync(const data::TensorView &src,
                                      data::TensorView *dst,
                                      void *cuda_stream = nullptr);
    };

} // namespace ptk::memory
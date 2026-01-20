#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>

#include "runtime/core/types.h"
#include "runtime/core/status.h"
#include "runtime/data/tensor.h"

//primarily used for cuda operations - save for now

namespace ptk::memory
{

    // Memory alignment requirements
    constexpr size_t kCpuAlignment = 64;     // AVX-512 alignment
    constexpr size_t kCudaAlignment = 256;   // NVIDIA recommended alignment
    constexpr size_t kDefaultAlignment = 32; // Minimum safe alignment

    // Helper to get alignment for device type
    inline size_t GetAlignmentForDevice(core::DeviceType device)
    {
        switch (device)
        {
        case core::DeviceType::kCpu:
            return kCpuAlignment;
        case core::DeviceType::kCuda:
            return kCudaAlignment;
        default:
            return kDefaultAlignment;
        }
    }

    // Deleter for CPU memory
    struct CpuDeleter
    {
        void operator()(void *ptr) const
        {
            if (ptr)
                std::free(ptr);
        }
    };

// Deleter for GPU memory (CUDA)
#ifndef __APPLE__
    struct CudaDeleter
    {
        void operator()(void *ptr) const;
    };
#endif

    // Ownership enum: who is responsible for freeing memory?
    enum class MemoryOwnership
    {
        kOwned,    // This object owns the memory and will delete it
        kBorrowed, // Memory is borrowed from elsewhere, don't delete
        kShared,   // Reference counted ownership (shared_ptr)
    };

    // Allocator interface for unified memory operations
    class Allocator
    {
    public:
        virtual ~Allocator() = default;

        // Allocate memory on specified device
        virtual core::Status Allocate(size_t num_bytes, core::DeviceType device,
                                      void **out_ptr) = 0;

        // Deallocate memory
        virtual core::Status Deallocate(void *ptr, core::DeviceType device) = 0;

        // Get device type this allocator serves
        virtual core::DeviceType GetDeviceType() const = 0;

        // Get name for logging/debugging
        virtual const char *GetName() const = 0;
    };

    // CPU memory allocator
    class CpuAllocator : public Allocator
    {
    public:
        CpuAllocator() = default;
        ~CpuAllocator() override = default;

        core::Status Allocate(size_t num_bytes, core::DeviceType device,
                              void **out_ptr) override;

        core::Status Deallocate(void *ptr, core::DeviceType device) override;

        core::DeviceType GetDeviceType() const override { return core::DeviceType::kCpu; }

        const char *GetName() const override { return "CpuAllocator"; }

    private:
        size_t AlignSize(size_t size) const;
    };

// GPU memory allocator (CUDA)
#ifndef __APPLE__
    class CudaAllocator : public Allocator
    {
    public:
        explicit CudaAllocator(int device_id = 0);
        ~CudaAllocator() override;

        core::Status Allocate(size_t num_bytes, core::DeviceType device,
                              void **out_ptr) override;

        core::Status Deallocate(void *ptr, core::DeviceType device) override;

        core::DeviceType GetDeviceType() const override { return core::DeviceType::kCuda; }

        const char *GetName() const override { return "CudaAllocator"; }

    private:
        int device_id_;
        size_t AlignSize(size_t size) const;
    };
#endif

    // Memory pool for reusable buffers
    class MemoryPool
    {
    public:
        explicit MemoryPool(Allocator *allocator);
        ~MemoryPool();

        // Allocate from pool (reuses if available)
        core::Status Allocate(size_t num_bytes, void **out_ptr);

        // Return buffer to pool for reuse
        core::Status Release(void *ptr, size_t num_bytes);

        // Get statistics
        size_t GetAllocatedBytes() const;
        size_t GetReservedBytes() const;

        // Clear all pooled buffers
        void Clear();

    private:
        struct PooledBuffer
        {
            void *ptr;
            size_t size;
        };

        Allocator *allocator_; // Not owned
        std::unordered_map<size_t, std::vector<PooledBuffer>> pool_;
        std::unordered_map<void *, size_t> allocated_; // Track what's allocated
    };

    // Unified tensor allocation API
    // Returns a TensorView with owned memory
    core::Status AllocateTensor(const data::TensorShape &shape, core::DataType dtype,
                                core::DeviceType device, Allocator *allocator,
                                data::TensorView *out_tensor);

    // Allocate tensor with ownership transfer
    core::Status AllocateTensorOwned(const data::TensorShape &shape, core::DataType dtype,
                                     core::DeviceType device, Allocator *allocator,
                                     std::unique_ptr<data::TensorView> *out_tensor);

    // Helper to calculate element size
    inline size_t GetElementSize(core::DataType dtype)
    {
        switch (dtype)
        {
        case core::DataType::kUint8:
            return 1;
        case core::DataType::kInt32:
            return 4;
        case core::DataType::kInt64:
            return 8;
        case core::DataType::kFloat32:
            return 4;
        case core::DataType::kFloat64:
            return 8;
        default:
            return 0;
        }
    }

    // Helper to calculate total bytes needed
    inline size_t CalculateTensorBytes(const data::TensorShape &shape,
                                       core::DataType dtype)
    {
        size_t total_elements = shape.num_elements();
        return total_elements * GetElementSize(dtype);
    }

} // namespace ptk::memory
#ifndef DYNBUFFER_HPP_INCLUDED
#define DYNBUFFER_HPP_INCLUDED

#include "resources.hpp"

namespace gpu {

  template <typename T>
  struct DynBuffer {
    DynBuffer(uint64_t alignment, uint32_t elements)
    : elems_count {elements}, block_size {get_block_size(sizeof(T), alignment)},
      buff {}
    {
      buff.create(VMA_MEMORY_USAGE_CPU_TO_GPU, elems_count * block_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    DynBuffer(DynBuffer<T> &&o) : elems_count {o.elems_count}, block_size {o.block_size}, buff {std::move(o.buff)} {}

    uint64_t get_offset(uint32_t element) const {
      return element * block_size;
    }

    T *get_mapped_ptr(uint32_t element) const {
      auto bytes = static_cast<uint8_t*>(buff.get_mapped_ptr());
      return reinterpret_cast<T*>(bytes + get_offset(element));
    }

    void flush() {
      buff.flush();
    }

    void flush(uint32_t element) {
      buff.flush(get_offset(element), block_size);
    }

    bool is_coherent() const {
      return buff.is_coherent();
    }

    VkBuffer api_buffer() const {
      return buff.get_api_buffer();
    }

    const DynBuffer<T> &operator=(DynBuffer<T> &&o) {
      std::swap(elems_count, o.elems_count);
      std::swap(block_size, o.block_size);
      buff = std::move(o.buff);
      return *this;
    }

  private:
    uint32_t elems_count;
    uint64_t block_size;
    gpu::Buffer buff;

    DynBuffer(const DynBuffer<T> &) = delete;
    const DynBuffer<T> &operator=(const DynBuffer<T> &) = delete;

    static uint64_t get_block_size(const uint64_t size, const uint64_t alignment) {
      if (!alignment) {
        return size;
      }

      auto mod = size % alignment;
      if (mod) {
        return size + alignment - mod;
      }
      return size;
    }  

  };

  template <typename T>
  struct UboBlock {
    T *ptr;
    uint32_t offset;
  };

  struct UniformBufferPool {
    UniformBufferPool(uint64_t alignment, uint64_t mem_size)
      : buffer {}, mem_alignment {alignment}
    {
      buffer.create(VMA_MEMORY_USAGE_CPU_TO_GPU, mem_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    VkBuffer api_buffer() const { return buffer.get_api_buffer(); }

    template<typename T>
    UboBlock<T> allocate_ubo() {
      auto val = allocate_chunk(sizeof(T));
      return {static_cast<T*>(val.ptr), val.offset};
    } 

    void reset() { write_offset = 0; }

    UniformBufferPool(UniformBufferPool &&) = default;
    UniformBufferPool &operator=(UniformBufferPool &&) = default; 
  private:
    gpu::Buffer buffer;
    uint64_t write_offset = 0;
    uint64_t mem_alignment = 0;

    UboBlock<void> allocate_chunk(uint64_t mem_size) {
      auto aligned_size = get_block_size(mem_size, mem_alignment);
      auto offset = write_offset;
      auto ptr = static_cast<uint8_t*>(buffer.get_mapped_ptr()) + offset;
      write_offset += aligned_size;
      return {static_cast<void*>(ptr), uint32_t(offset)};
    }

    static uint64_t get_block_size(const uint64_t size, const uint64_t alignment) {
      if (!alignment) {
        return size;
      }

      auto mod = size % alignment;
      if (mod) {
        return size + alignment - mod;
      }
      return size;
    }

  };

}


#endif
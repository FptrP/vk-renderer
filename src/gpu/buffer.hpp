#ifndef BUFFER_HPP_INCLUDED
#define BUFFER_HPP_INCLUDED

#include "init.hpp"

namespace gpu {

  struct Buffer {
    Buffer(Device &dev) : base {dev} {}
    ~Buffer() { close(); }
    
    void create(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage);
    void close();
    void flush(uint64_t offset = 0, uint64_t size = VK_WHOLE_SIZE);

    VkBuffer get_api_buffer() const { return handle; }
    uint64_t get_size() const { return size; }
    bool is_coherent() const { return coherent; }
    void *get_mapped_ptr() const { return mapped_ptr; }

    Buffer(Buffer&) = delete;
    const Buffer &operator=(const Buffer&) = delete;
  
    Buffer(Buffer &&o)
      : base {o.base}, handle {o.handle}, 
        allocation {o.allocation}, size {o.size}, coherent {o.coherent}, mapped_ptr {o.mapped_ptr} 
    {
      o.handle = nullptr;
    }

    const Buffer &operator=(Buffer &o) {
      std::swap(base, o.base);
      std::swap(handle, o.handle);
      std::swap(allocation, o.allocation);
      std::swap(size, o.size);
      std::swap(coherent, o.coherent);
      std::swap(mapped_ptr, o.mapped_ptr);
      return *this;
    }

  private:
    Device &base;
    VkBuffer handle {nullptr};
    VmaAllocation allocation {nullptr};
    uint64_t size {0};
    bool coherent = false;
    void *mapped_ptr = nullptr;
  };

}

#endif
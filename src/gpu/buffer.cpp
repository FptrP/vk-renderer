#include "buffer.hpp"

namespace gpu {
  void Buffer::create(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage) {
    close();

    VkBufferCreateInfo buffer_info {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = buffer_size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr
    };

    VmaAllocationCreateInfo alloc_info {};
    alloc_info.usage = memory;
    alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VmaAllocationInfo info {};
    
    VKCHECK(vmaCreateBuffer(base.get_allocator(), &buffer_info, &alloc_info, &handle, &allocation, &info));
    
    VkMemoryPropertyFlags mem_flags = 0;
    vmaGetMemoryTypeProperties(base.get_allocator(), info.memoryType, &mem_flags);
    coherent = mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; 
    mapped_ptr = info.pMappedData;
    size = buffer_size;
  }
  
  void Buffer::close() {
    if (!handle) return;

    vmaDestroyBuffer(base.get_allocator(), handle, allocation);
    handle = nullptr;
    allocation = nullptr;
    size = 0;
    mapped_ptr = nullptr;
    coherent = false;
  }

  void Buffer::flush(uint64_t offset, uint64_t size) {
    if (coherent) return;
    vmaFlushAllocation(base.get_allocator(), allocation, offset, size);
  }
}
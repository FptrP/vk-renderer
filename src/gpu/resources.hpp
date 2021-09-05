#ifndef RESOURCES_HPP_INCLUDED
#define RESOURCES_HPP_INCLUDED

#include "vkerror.hpp"
#include <lib/vk_mem_alloc.h>

#include <algorithm>
#include <unordered_map>

namespace gpu {
  struct ImageViewRange {
    VkImageViewType type;
    VkImageSubresourceRange range;

    bool operator==(const ImageViewRange &o) const {
      return (type == o.type)
        && (range.aspectMask == o.range.aspectMask)
        && (range.baseArrayLayer == o.range.baseArrayLayer)
        && (range.baseMipLevel == o.range.baseMipLevel)
        && (range.layerCount == o.range.layerCount)
        && (range.levelCount == o.range.levelCount); 
    }
  };
}

namespace std {
  template <>
  struct hash<gpu::ImageViewRange> {

    size_t operator()(const gpu::ImageViewRange &key) const {
      size_t h = 0;
      h ^= size_t(key.type);
      h ^= size_t(key.range.aspectMask) << 1;
      h ^= key.range.baseArrayLayer << 2;
      h ^= key.range.baseMipLevel << 3;
      h ^= key.range.layerCount << 4;
      h ^= key.range.levelCount << 5;
      return h;  
    }
  };
}

namespace gpu {

  struct Buffer {
    Buffer(VmaAllocator alloc) : base {alloc} {}
    ~Buffer() { close(); }
    
    void create(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage);
    void close();
    void flush(uint64_t offset = 0, uint64_t size = VK_WHOLE_SIZE);

    VkBuffer get_api_buffer() const { return handle; }
    uint64_t get_size() const { return size; }
    bool is_coherent() const { return coherent; }
    bool is_empty() const { return (size == 0) || (handle == nullptr); }
    
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
    VmaAllocator base {nullptr}; 
    VkBuffer handle {nullptr};
    VmaAllocation allocation {nullptr};
    uint64_t size {0};
    bool coherent = false;
    void *mapped_ptr = nullptr;
  };

  struct ImageInfo {
    VkFormat format;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;

    const VkExtent3D extent3D() const { return {width, height, depth}; }
    const VkExtent2D extent2D() const { return {width, height}; }
  };

  struct Image {
    Image(VkDevice dev, VmaAllocator alloc) : device {dev}, allocator {alloc} {}
    ~Image() { close(); }

    void create(VkImageType type, VkFormat fmt, VkExtent3D ext, uint32_t mips, uint32_t layers, VkImageTiling tiling, VkImageUsageFlags usage);
    void create_reference(VkImage image, const ImageInfo &info);

    void close();
    VkImageView get_view(const ImageViewRange &key);

    VkImage get_image() const { return handle; }
    VkExtent3D get_extent() const { return descriptor.extent3D(); }

    uint32_t get_mip_levels() const { return descriptor.mip_levels; }
    uint32_t get_array_layers() const { return descriptor.array_layers; }

    Image(Image &&o) 
      : device {o.device}, allocator {o.allocator}, handle {o.handle},
        allocation {o.allocation}, descriptor{o.descriptor}, views{std::move(o.views)}
    {
      o.handle = nullptr;
    }

    const Image &operator=(Image&& i) {
      std::swap(device, i.device);
      std::swap(allocator, i.allocator);
      std::swap(handle, i.handle);
      std::swap(allocation, i.allocation);
      std::swap(descriptor, i.descriptor);
      std::swap(views, i.views);
      return *this;
    }

  private:
    VkDevice device {nullptr};
    VmaAllocator allocator {nullptr};
    VkImage handle {nullptr};
    VmaAllocation allocation {nullptr};
    ImageInfo descriptor {};
    std::unordered_map<ImageViewRange, VkImageView> views;
  };

}

#endif
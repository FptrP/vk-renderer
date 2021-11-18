#ifndef RESOURCES_HPP_INCLUDED
#define RESOURCES_HPP_INCLUDED

#include "common.hpp"
#include "driver.hpp"
#include <lib/vk_mem_alloc.h>

#include <algorithm>
#include <unordered_map>

namespace gpu {
  struct ImageViewRange {
    VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;
    VkImageAspectFlags aspect = 0;
    uint32_t base_mip = 0;
    uint32_t mips_count = 1;
    uint32_t base_layer = 0;
    uint32_t layers_count = 1;
    
    ImageViewRange() {}

    ImageViewRange(VkImageViewType type_, VkImageAspectFlags aspect_, uint32_t base_mip_, uint32_t mips_count_, uint32_t base_layer_, uint32_t layers_count_)
      : type {type_}, aspect {aspect_}, base_mip {base_mip_}, mips_count {mips_count_}, base_layer {base_layer_}, layers_count {layers_count_} {}
    
    ImageViewRange(VkImageViewType type_, uint32_t base_mip_, uint32_t mips_count_, uint32_t base_layer_, uint32_t layers_count_)
      : ImageViewRange(type_, 0, base_mip_, mips_count_,  base_layer_, layers_count_) {}


    bool operator==(const ImageViewRange &o) const {
      return (type == o.type)
        && (aspect == o.aspect)
        && (base_mip == o.base_mip)
        && (mips_count == o.mips_count)
        && (base_layer == o.base_layer)
        && (layers_count == o.layers_count); 
    }
  };
}

namespace std {
  template <>
  struct hash<gpu::ImageViewRange> {

    size_t operator()(const gpu::ImageViewRange &key) const {
      size_t h = 0;
      h ^= size_t(key.type);
      h ^= key.aspect << 1;
      h ^= key.base_layer << 2;
      h ^= key.layers_count << 3;
      h ^= key.base_mip << 4;
      h ^= key.mips_count << 5;
      return h;  
    }
  };
}

namespace gpu {

  struct Buffer {
    Buffer() {}
    ~Buffer() { close(); }
    
    void create(VmaMemoryUsage memory, uint64_t buffer_size, VkBufferUsageFlags usage);
    void close();
    void flush(uint64_t offset = 0, uint64_t size = VK_WHOLE_SIZE);

    VkBuffer get_api_buffer() const { return handle; }
    uint64_t get_size() const { return size; }
    bool is_coherent() const { return coherent; }
    bool is_empty() const { return (size == 0) || (handle == nullptr); }
    
    void *get_mapped_ptr() const { return mapped_ptr; }

    VkDeviceAddress get_device_address() const;

    Buffer(Buffer&) = delete;
    const Buffer &operator=(const Buffer&) = delete;
  
    Buffer(Buffer &&o)
      : handle {o.handle}, 
        allocation {o.allocation}, size {o.size}, coherent {o.coherent}, mapped_ptr {o.mapped_ptr} 
    {
      o.handle = nullptr;
    }

    Buffer &operator=(Buffer &&o) {
      std::swap(handle, o.handle);
      std::swap(allocation, o.allocation);
      std::swap(size, o.size);
      std::swap(coherent, o.coherent);
      std::swap(mapped_ptr, o.mapped_ptr);
      return *this;
    }

  private:
    VkBuffer handle {nullptr};
    VmaAllocation allocation {nullptr};
    uint64_t size {0};
    bool coherent = false;
    void *mapped_ptr = nullptr;
  };

  struct ImageInfo {
    VkFormat format;
    VkImageAspectFlags aspect = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;

    ImageInfo() {}
    ImageInfo(VkFormat fmt, VkImageAspectFlags aspect_flags, uint32_t w, uint32_t h)
      : format {fmt}, aspect {aspect_flags}, width {w}, height {h} {}
    
    ImageInfo(VkFormat fmt, VkImageAspectFlags aspect_flags, uint32_t w, uint32_t h, uint32_t d, uint32_t mips, uint32_t layers)
      : format {fmt}, aspect {aspect_flags}, width {w}, height {h}, depth {d}, mip_levels{mips}, array_layers{layers} {}


    const VkExtent3D extent3D() const { return {width, height, depth}; }
    const VkExtent2D extent2D() const { return {width, height}; }
  };

  struct Image {
    Image() {}
    ~Image() { close(); }

    void create(VkImageType type, const ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage);
    void create_reference(VkImage image, const ImageInfo &info);

    void close();
    VkImageView get_view(ImageViewRange key);

    VkImage get_image() const { return handle; }
    VkExtent3D get_extent() const { return descriptor.extent3D(); }
    VkFormat get_fmt() const { return descriptor.format; }
    uint32_t get_mip_levels() const { return descriptor.mip_levels; }
    uint32_t get_array_layers() const { return descriptor.array_layers; }
    const ImageInfo &get_info() const { return descriptor; }

    Image(Image &&o) 
      : handle {o.handle},
        allocation {o.allocation}, descriptor{o.descriptor}, views{std::move(o.views)}
    {
      o.handle = nullptr;
      o.views.clear();
    }

    Image &operator=(Image&& i) {
      std::swap(handle, i.handle);
      std::swap(allocation, i.allocation);
      std::swap(descriptor, i.descriptor);
      std::swap(views, i.views);
      return *this;
    }

  private:
    VkImage handle {nullptr};
    VmaAllocation allocation {nullptr};
    ImageInfo descriptor {};
    std::unordered_map<ImageViewRange, VkImageView> views;
  };

  struct AccelerationStructure {

  };
}

#endif
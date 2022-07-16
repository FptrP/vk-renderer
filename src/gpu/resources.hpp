#ifndef RESOURCES_HPP_INCLUDED
#define RESOURCES_HPP_INCLUDED

#include "common.hpp"
#include "driver.hpp"
#include "resource_info.hpp"

#include <lib/vk_mem_alloc.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace gpu {

  enum class ImageCreateOptions {
    None,
    Cubemap,
    Array2D
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

    void create(VkImageType type, const ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage, ImageCreateOptions options = ImageCreateOptions::None);
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
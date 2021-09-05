#ifndef IMAGE_HPP_INCLUDED
#define IMAGE_HPP_INCLUDED

#include "init.hpp"
#include <unordered_map>

namespace gpu {

struct ImageViewKey {
  ImageViewKey() {}
  ImageViewKey &set_type(VkImageViewType vtype) {
    mask &= ~(VTYPE_MASK << VTYPE_SHIFT);
    mask |= (uint64_t(vtype) & VTYPE_MASK) << VTYPE_SHIFT;
    return *this;
  }
  
  ImageViewKey &set_aspect(VkImageAspectFlags flags) {
    mask &= ~(ASPECT_MASK << ASPECT_SHIFT);
    mask |= (uint64_t(flags) & ASPECT_MASK) << ASPECT_SHIFT;
    return *this;
  }

  ImageViewKey &set_range(uint8_t base_mip, uint8_t mip_count, uint16_t base_level, uint16_t level_count) {
    mask &= ~(BASE_MIP_MASK << BASE_MIP_SHIFT)|(MIP_COUNT_MASK << MIP_COUNT_SHIFT)|(BASE_LEVEL_MASK << BASE_LEVEL_SHIFT)|(LEVEL_COUNT_MASK << LEVEL_COUNT_SHIFT);
    mask |= (uint64_t(base_mip) & BASE_MIP_MASK) << BASE_MIP_SHIFT;
    mask |= (uint64_t(mip_count) & MIP_COUNT_MASK) << MIP_COUNT_SHIFT;
    mask |= (uint64_t(base_level) & BASE_LEVEL_MASK) << BASE_LEVEL_SHIFT;
    mask |= (uint64_t(level_count) & LEVEL_COUNT_MASK) << LEVEL_COUNT_SHIFT;
    return *this;
  }

  uint64_t get_mask() const {
    return mask;
  }

  VkImageViewType get_type() const {
    uint64_t vtype = (mask >> VTYPE_SHIFT) & VTYPE_MASK;
    return VkImageViewType(vtype);
  }  

  VkImageAspectFlags get_aspect_flags() const {
    return VkImageAspectFlags((mask >> ASPECT_SHIFT) & ASPECT_MASK);
  }

  uint8_t get_base_mip() const {
    return (mask >> BASE_MIP_SHIFT) & BASE_MIP_MASK;
  }

  uint8_t get_mip_count() const {
    return (mask >> MIP_COUNT_SHIFT) & MIP_COUNT_MASK;
  }

  uint16_t get_base_level() const {
    return (mask >> BASE_LEVEL_SHIFT) & BASE_LEVEL_MASK;
  }

  uint16_t get_level_count() const {
    return (mask >> LEVEL_COUNT_SHIFT) & LEVEL_COUNT_MASK;
  }

private:
  static constexpr uint64_t VTYPE_SHIFT = 56;
  static constexpr uint64_t VTYPE_MASK = 0xff;
  static constexpr uint64_t ASPECT_SHIFT = 48;
  static constexpr uint64_t ASPECT_MASK = 0xff;
  static constexpr uint64_t BASE_MIP_SHIFT = 40;
  static constexpr uint64_t BASE_MIP_MASK = 0xff;
  static constexpr uint64_t MIP_COUNT_SHIFT = 32;
  static constexpr uint64_t MIP_COUNT_MASK = 0xff;
  static constexpr uint64_t BASE_LEVEL_SHIFT = 16;
  static constexpr uint64_t BASE_LEVEL_MASK = 0xffff;
  static constexpr uint64_t LEVEL_COUNT_SHIFT = 0;
  static constexpr uint64_t LEVEL_COUNT_MASK = 0xffff;

  uint64_t mask {0};
};

struct ImageInfo {
  VkFormat format;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t mip_levels;
  uint32_t array_layers;

  const VkExtent3D extent3D() const { return {width, height, depth}; }
  const VkExtent2D extent2D() const { return {width, height}; }
};

struct Image {
  Image(Device &device) : base {device} {}
  ~Image() { close(); }

  void create(VkImageType type, VkFormat fmt, VkExtent3D ext, uint32_t mips, uint32_t layers, VkImageTiling tiling, VkImageUsageFlags usage);
  void create_reference(VkImage image, const VkImageCreateInfo &info);

  void close();
  VkImageView get_view(const ImageViewKey key);

  VkImage get_image() const { return handle; }
  VkExtent3D get_extent() const { return image_info.extent; }
  VkImageUsageFlags get_usage() const { return image_info.usage; }

  uint32_t get_mip_levels() const { return image_info.mipLevels; }
  uint32_t get_array_layers() const { return image_info.arrayLayers; }

  Image(Image &&o) 
    : base {o.base}, is_reference {o.is_reference}, image_info{o.image_info}, handle {o.handle}, allocation {o.allocation}, views{std::move(o.views)}
  {
    o.handle = nullptr;
  }

  const Image &operator=(Image&& i) {
    std::swap(is_reference, i.is_reference);
    std::swap(base, i.base);
    std::swap(image_info, i.image_info);
    std::swap(handle, i.handle);
    std::swap(allocation, i.allocation);
    std::swap(views, i.views);
    return *this;
  }

private:
  Image(Image &) = delete;
  const Image &operator=(const Image&) = delete;
  
  Device &base;
  bool is_reference = false;
  VkImageCreateInfo image_info {};

  VkImage handle {nullptr};
  VmaAllocation allocation {nullptr};
  std::unordered_map<uint64_t, VkImageView> views;
};


}

#endif
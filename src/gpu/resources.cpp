#include "resources.hpp"
#include <stdexcept>

namespace gpu {
  static VkImageCreateFlags options_to_flags(ImageCreateOptions options) {
    switch (options) {
    case ImageCreateOptions::Array2D:
      return VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    case ImageCreateOptions::Cubemap:
      return VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    default:
      return 0;
    }
  }

  void Image::create(VkImageType type, const ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage, ImageCreateOptions options) {
    close();
    auto allocator = app_device().get_allocator();

    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = options_to_flags(options),
      .imageType = type,
      .format = info.format,
      .extent = info.extent3D(),
      .mipLevels = info.mip_levels,
      .arrayLayers = info.array_layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };
    
    VmaAllocationCreateInfo alloc_info {};
    alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    descriptor = info;
    VKCHECK(vmaCreateImage(allocator, &image_info, &alloc_info, &handle, &allocation, nullptr));
  }
  
  void Image::create_reference(VkImage image, const ImageInfo &info) {
    close();
    allocation = nullptr;
    handle = image;
    descriptor = info;
  }

  void Image::close() {
    auto allocator = app_device().get_allocator();
    auto device = app_device().api_device();

    for (const auto &pair : views) {
      vkDestroyImageView(device, pair.second, nullptr);
    }

    views.clear();

    if (handle && allocation) {
      vmaDestroyImage(allocator, handle, allocation);
      handle = nullptr;
      allocation = nullptr;
    }
  }
  

  VkImageView Image::get_view(ImageViewRange range) {
    auto device = app_device().api_device();

    range.aspect &= descriptor.aspect;
    if (!range.aspect) {
      range.aspect = descriptor.aspect;
    }

    auto iter = views.find(range);
    if (iter != views.end()) {
      return iter->second;
    }

    VkImageViewCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = handle,
      .viewType = range.type,
      .format = descriptor.format,
      .components = {},
      .subresourceRange = {range.aspect, range.base_mip, range.mips_count, range.base_layer, range.layers_count}
    };

    VkImageView view {nullptr};
    VKCHECK(vkCreateImageView(device, &info, nullptr, &view));
    views.insert({range, view});
    return view;
  }

}
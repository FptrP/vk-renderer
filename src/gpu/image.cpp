#include "image.hpp"

namespace gpu {

  void Image::create(VkImageType type, VkFormat fmt, VkExtent3D ext, uint32_t mips, uint32_t layers, VkImageTiling tiling, VkImageUsageFlags usage) {
    close();
    is_reference = false;

    VkImageCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = type,
      .format = fmt,
      .extent = ext,
      .mipLevels = mips,
      .arrayLayers = layers,
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

    image_info = info;
    VKCHECK(vmaCreateImage(base.get_allocator(), &info, &alloc_info, &handle, &allocation, nullptr));
  }
  
  void Image::create_reference(VkImage image, const VkImageCreateInfo &info) {
    image_info = info;
    is_reference = true;
    allocation = nullptr;
    handle = image;
  }

  void Image::close() {
    for (const auto &pair : views) {
      vkDestroyImageView(base.get_device(), pair.second, nullptr);
    }

    if (handle && !is_reference) {
      vmaDestroyImage(base.get_allocator(), handle, allocation);
      handle = nullptr;
    }
  }
  
  VkImageView Image::get_view(const ImageViewKey key) {
    auto iter = views.find(key.get_mask());
    if (iter != views.end()) {
      return iter->second;
    }

    VkImageViewCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = handle,
      .viewType = key.get_type(),
      .format = image_info.format,
      .components = {},
      .subresourceRange = {key.get_aspect_flags(), key.get_base_mip(), key.get_mip_count(), key.get_base_level(), key.get_level_count()}
    };

    VkImageView view {nullptr};
    VKCHECK(vkCreateImageView(base.get_device(), &info, nullptr, &view));
    views.insert({key.get_mask(), view});
    return view;
  }

}
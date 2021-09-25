#include "resources.hpp"
#include <stdexcept>

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
    
    VKCHECK(vmaCreateBuffer(base, &buffer_info, &alloc_info, &handle, &allocation, &info));
    
    VkMemoryPropertyFlags mem_flags = 0;
    vmaGetMemoryTypeProperties(base, info.memoryType, &mem_flags);
    coherent = mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; 
    mapped_ptr = info.pMappedData;
    size = buffer_size;
  }
  
  void Buffer::close() {
    if (!handle) return;

    vmaDestroyBuffer(base, handle, allocation);
    base = nullptr;
    handle = nullptr;
    allocation = nullptr;
    size = 0;
    mapped_ptr = nullptr;
    coherent = false;
  }

  void Buffer::flush(uint64_t offset, uint64_t size) {
    if (coherent) return;
    vmaFlushAllocation(base, allocation, offset, size);
  }

  void Image::create(VkImageType type, const ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage) {
    close();

    VkImageCreateInfo image_info {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
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
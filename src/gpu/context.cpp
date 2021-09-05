#include "context.hpp"

namespace gpu {



  void ResourceState::image_barrier(
    const gpu::Image &image,
    VkPipelineStageFlags stages,
    VkAccessFlags dst_access,
    VkImageLayout dst_layout,
    VkImageSubresourceRange dst_range)
  {
    //acquire an image or create with UNDEFINDED layout

    auto it = image_states.find(image.get_image());
    if (it == image_states.end()) {
      it = image_states.emplace(image.get_image(), ImageState{image.get_array_layers(), image.get_mip_levels()}).first;
    }

    ImageState &state = it->second;

    //split range by same src layout 
    const auto final_layer = dst_range.baseArrayLayer + dst_range.layerCount;
    const auto final_mip = dst_range.baseMipLevel + dst_range.levelCount; 

    VkImageMemoryBarrier next_barrier {};
    next_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    next_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    next_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    next_barrier.image = image.get_image();

    for (auto layer = dst_range.baseArrayLayer; layer < final_layer; layer++) {
      for (auto level = dst_range.baseMipLevel; level < final_mip; level++) {
        

        auto &subresource = state.subresources[layer * image.get_mip_levels() + level];

        next_barrier.srcAccessMask = subresource.access;
        next_barrier.dstAccessMask = dst_access;
        next_barrier.oldLayout = subresource.layout;
        next_barrier.newLayout = dst_layout;
        next_barrier.subresourceRange = {dst_range.aspectMask, level, 1, layer, 1};

        image_barriers.push_back(next_barrier);

        src_stages |= subresource.stages;

        subresource.stages = stages;
        subresource.access = dst_access;
        subresource.layout = dst_layout;
      }
    }

    dst_stages |= stages;
  }
    
  void ResourceState::buffer_barrier(VkBuffer buffer, VkPipelineStageFlags dst_stages, VkAccessFlags dst_access)
  {

  }

  void ResourceState::flush(VkCommandBuffer cmd) {
    if (src_stages == 0 && dst_stages == 0) {
      return;
    }

    src_stages = src_stages? src_stages : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_stages = dst_stages? dst_stages : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    vkCmdPipelineBarrier(
      cmd,
      src_stages,
      dst_stages,
      0,
      mem_barriers.size(),
      mem_barriers.data(),
      0,
      nullptr,
      image_barriers.size(),
      image_barriers.data());  

    src_stages = 0;
    dst_stages = 0;
    
    image_barriers.clear();
    mem_barriers.clear();
  }


}
#ifndef CONTEXT_HPP_INCLUDED
#define CONTEXT_HPP_INCLUDED

#include "init.hpp"
#include "image.hpp"

#include <unordered_map>

namespace gpu {

  struct ResourceUsage {
    VkPipelineStageFlags stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags access = 0;
  };

  struct ImageSubresource : ResourceUsage {
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
  };

  struct ImageState {
    ImageState() {}
    ImageState(uint32_t layers, uint32_t levels) : array_layers{layers}, mip_levels {levels} {
      subresources.reset(new ImageSubresource[array_layers * mip_levels]);
    }

    uint32_t array_layers = 0;
    uint32_t mip_levels = 0;
    std::unique_ptr<ImageSubresource[]> subresources; 
  };

  struct ResourceState {

    void image_barrier(
      const gpu::Image &image,
      VkPipelineStageFlags dst_stages,
      VkAccessFlags dst_access,
      VkImageLayout dst_layout,
      VkImageSubresourceRange dst_range);
    
    void buffer_barrier(VkBuffer buffer, VkPipelineStageFlags dst_stages, VkAccessFlags dst_access);

    void flush(VkCommandBuffer cmd);

  private:
    VkPipelineStageFlags dst_stages = 0;
    VkPipelineStageFlags src_stages = 0;
    std::vector<VkMemoryBarrier> mem_barriers;
    std::vector<VkImageMemoryBarrier> image_barriers;

    std::unordered_map<VkBuffer, ResourceUsage> buffer_states;
    std::unordered_map<VkImage, ImageState> image_states;
  };

}

#endif
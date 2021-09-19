#include "rendergraph.hpp"

namespace rendergraph {
  
  static VkPipelineStageFlags get_pipeline_flags(VkShaderStageFlags stages) {
    VkPipelineStageFlags pipeline_stages = 0;
    if (stages & VK_SHADER_STAGE_VERTEX_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_GEOMETRY_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
    }
    if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
      pipeline_stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    return pipeline_stages;
  }


  ImageViewId RenderGraphBuilder::use_color_attachment(ImageResourceId id, uint32_t mip, uint32_t layer) {
    ImageSubresourceId subres {id, mip, layer};
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    
    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
    }
    auto &src = input.images[subres];
    if (src.layout != state.layout || src.access != state.access || src.stages != state.stages) {
      throw std::runtime_error {"Incompatible image usage"};
    }

    return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
  }
  
  ImageViewId RenderGraphBuilder::use_depth_attachment(ImageResourceId id, uint32_t mip, uint32_t layer) {
    ImageSubresourceId subres {id, mip, layer};
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };
    
    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
    }
    auto &src = input.images[subres];
    if (src.layout != state.layout || src.access != state.access || src.stages != state.stages) {
      throw std::runtime_error {"Incompatible image usage"};
    }
    return ImageViewId {id, gpu::ImageViewRange {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
  }
  
  ImageViewId RenderGraphBuilder::sample_image(ImageResourceId id, VkShaderStageFlags stages, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
    auto pipeline_stages = get_pipeline_flags(stages);

    ImageSubresourceState state {
      pipeline_stages,
      VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };

    for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
      for (uint32_t mip = base_mip; mip < base_mip + mip_count; mip++) {
        ImageSubresourceId subres {id, mip, layer};
        if (!input.images.count(subres)) {        
          input.images[subres] = state;
          continue;
        }

        auto &src = input.images[subres];
        if (state.layout != src.layout) {
          throw std::runtime_error {"Incompatible layout"};
        }

        src.stages |= state.stages;
        src.access |= state.access;
      }
    }

    return ImageViewId {id, {VK_IMAGE_VIEW_TYPE_2D, base_mip, mip_count, base_layer, layer_count}};
  }

  void RenderGraphBuilder::prepare_backbuffer() {
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    ImageSubresourceId subres {backbuffer, 0, 0};
    
    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return;
    }

    throw std::runtime_error {"Incompatible access for backbuffer"};
  }

  ImageViewId RenderGraphBuilder::use_backbuffer_attachment() {

    ImageSubresourceState state {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    ImageSubresourceId subres {backbuffer, 0, 0};

    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return {backbuffer, gpu::ImageViewRange{VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1}};
    }

    throw std::runtime_error {"Incompatible access for backbuffer"};
    return {backbuffer, {VK_IMAGE_VIEW_TYPE_2D, 0, 1, 0, 1}};
  }

  
  const gpu::ImageInfo &RenderGraphBuilder::get_image_info(ImageResourceId id) {
    return resources.get_info(id);
  }
  
  RenderGraph::RenderGraph(gpu::Device &device, gpu::Swapchain &swapchain)
    : gpu {device, swapchain},
      resources {device.get_allocator(), device.api_device()}
  {
    auto vk_backbuffers = gpu.take_backbuffers();
    backbuffers.reserve(vk_backbuffers.size());

    for (auto &img : vk_backbuffers) {
      backbuffers.push_back(resources.create_global_image_ref(img));
    }
    gpu.acquire_image();

    auto index = gpu.get_backbuf_index();
    if (index != 0) {
      resources.remap(backbuffers[0], backbuffers[index]);
    }
  }

  RenderGraph::~RenderGraph() {
    vkDeviceWaitIdle(gpu.get_device().api_device());
  }

  void RenderGraph::submit() {
    tracking_state.flush(resources);
    //tracking_state.dump_barriers();
    auto barriers = tracking_state.take_barriers();
    tracking_state.clear();

    gpu.begin();

    auto &api_cmd = gpu.get_cmdbuff(); 
    RenderResources res {resources, gpu};

    for (uint32_t i = 0; i < tasks.size(); i++) {
      if (barriers.size() > i) {
        write_barrier(barriers[i], api_cmd.get_command_buffer());
      }
      tasks[i]->write_commands(res, api_cmd);
      api_cmd.end_renderpass(); //to be sure about barriers
    }

    gpu.submit();

    tasks.clear();

    if (gpu.get_backbuf_index() != 0) {
      resources.remap(backbuffers[0], backbuffers[gpu.get_backbuf_index()]);
    }

    gpu.acquire_image();

    auto backbuffer_index = gpu.get_backbuf_index(); 
    if (backbuffer_index != 0) {
      resources.remap(backbuffers[0], backbuffers[backbuffer_index]);
    }

  }

  ImageResourceId RenderGraph::create_image(VkImageType type, const gpu::ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage) {
    return resources.create_global_image(ImageDescriptor {
      type,
      info.format,
      info.aspect,
      tiling,
      usage,
      info.width,
      info.height,
      info.depth,
      info.mip_levels,
      info.array_layers
    });
  }

  ImageResourceId RenderGraph::create_image(const ImageDescriptor &desc) {
    return resources.create_global_image(desc);
  }

  ImageResourceId RenderGraph::get_backbuffer() const {
    return backbuffers[0];
  }

  void RenderGraph::write_barrier(const Barrier &barrier, VkCommandBuffer cmd) {
    if (barrier.is_empty()) {
      return;
    }
    
    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkMemoryBarrier> mem_barriers;

    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;

    for (const auto &state : barrier.image_barriers) {
      auto &image = resources.get_image(state.id.id);
      const auto &desc = resources.get_info(state.id.id);
      
      if (state.id.mip >= desc.mip_levels || state.id.layer >= desc.array_layers) {
        throw std::runtime_error {"Image subresource out of range"};
      }
      
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkImageMemoryBarrier img_barrier {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access,
        state.src.layout,
        state.dst.layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image.get_image(),
        {desc.aspect, state.id.mip, 1, state.id.layer, 1}
      };
      image_barriers.push_back(img_barrier);
    }

    for (const auto &state : barrier.buffer_barriers) {
      src_stages |= state.src.stages;
      dst_stages |= state.dst.stages;

      VkMemoryBarrier mem_barrier {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        state.src.access,
        state.dst.access
      };
      mem_barriers.push_back(mem_barrier);
    }

    if (!src_stages) {
      src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    vkCmdPipelineBarrier(cmd, 
      src_stages, 
      dst_stages, 
      0, 
      mem_barriers.size(), 
      mem_barriers.data(), 
      0,
      nullptr,
      image_barriers.size(),
      image_barriers.data());
  }

  gpu::Buffer &RenderResources::get_buffer(BufferResourceId id) {
    return resources.get_buffer(id);
  }
  
  gpu::Image &RenderResources::get_image(ImageResourceId id) {
    return resources.get_image(id);
  }
  
  VkImageView RenderResources::get_view(const ImageViewId &ref) {
    return get_image(ref.get_id()).get_view(ref.get_range());
  }

}
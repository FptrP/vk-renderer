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


  void RenderGraphBuilder::use_color_attachment(std::size_t image_id, uint32_t mip, uint32_t layer) {
    ImageSubresource subres {image_id, mip, layer};
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    
    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return;
    }
    auto &src = input.images[subres];
    if (src.layout != state.layout || src.access != state.access || src.stages != state.stages) {
      throw std::runtime_error {"Incompatible image usage"};
    }
  }
  
  void RenderGraphBuilder::use_depth_attachment(std::size_t image_id, uint32_t mip, uint32_t layer) {
    ImageSubresource subres {image_id, mip, layer};
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };
    
    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return;
    }
    auto &src = input.images[subres];
    if (src.layout != state.layout || src.access != state.access || src.stages != state.stages) {
      throw std::runtime_error {"Incompatible image usage"};
    }
  }
  
  void RenderGraphBuilder::sample_image(std::size_t image_id, VkShaderStageFlags stages, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
    auto pipeline_stages = get_pipeline_flags(stages);

    ImageSubresourceState state {
      pipeline_stages,
      VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };

    for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
      for (uint32_t mip = base_mip; mip < base_mip + mip_count; mip++) {
        ImageSubresource subres {image_id, mip, layer};
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
  }

  void RenderGraphBuilder::prepare_backbuffer() {
    auto hash = get_backbuffer_hash();
    ImageSubresourceState state {
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };
    ImageSubresource subres {hash, 0, 0};
    
    if (!input.images.count(subres)) {
      input.images[subres] = state;
      return;
    }

    throw std::runtime_error {"Incompatible access for backbuffer"};
  }

  void RenderGraphBuilder::create_img(std::size_t hash, const ImageDescriptor &desc) {
    if (!resources.image_remap.count(hash)) {
      auto index = resources.images.size();
      auto img = gpu.get_device().new_image();
      img.create(desc.type, desc.get_vk_info(), desc.tiling, desc.usage);
      resources.images.push_back(std::move(img));
      resources.image_remap[hash] = index;
      return;
    }

    throw std::runtime_error {"Attempt to recreate image"};
  }

  void RenderGraphBuilder::create_buf(std::size_t hash, const BufferDescriptor &desc) {
    if (!resources.buffer_remap.count(hash)) {
      auto index = resources.buffers.size();
      auto buf = gpu.get_device().new_buffer();
      buf.create(desc.memory_type, desc.size, desc.usage);
      resources.buffers.push_back({std::move(buf), {}});
      resources.buffer_remap[hash] = index;
    }

    throw std::runtime_error {"Attemp to recreate buffer"};
  }

  
  RenderGraph::RenderGraph(gpu::Device &device, gpu::Swapchain &swapchain)
    : gpu {device, swapchain}
  {
    auto backbuffers = gpu.take_backbuffers();
    resources.images.reserve(backbuffers.size());

    for (uint32_t i = 0; i < backbuffers.size(); i++) {
      resources.images.push_back(Image {std::move(backbuffers[i])});
    }
  }

  void RenderGraph::submit() {
    if (tracking_state.is_dirty()) {
      tracking_state.flush();
      tracking_state.dump_barriers();
    }

    gpu.begin();
    remap_backbuffer();

    auto api_cmd = gpu.get_cmdbuff(); 
    auto &barriers = tracking_state.get_barriers();
    RenderResources res {resources, gpu};

    for (uint32_t i = 0; i < tasks.size(); i++) {
      if (barriers.size() > i) {
        write_barrier(barriers[i], api_cmd);
      }
      tasks[i]->write_commands(res, api_cmd);
    }

    gpu.submit();
    tracking_state.set_external_state(resources);
  }

  void RenderGraph::remap_backbuffer() {
    resources.image_remap[get_backbuffer_hash()] = gpu.get_backbuf_index();
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
      auto &image = resources.images.at(resources.image_remap.at(state.image_hash));
      const auto &desc = image.vk_image.get_info();
      
      if (state.mip >= desc.mip_levels || state.layer >= desc.array_layers) {
        throw std::runtime_error {"Image subresource out of range"};
      }
      
      auto src_state = state.src;
      if (state.acquire_barrier) {
        src_state = image.get_external_state({0, state.layer, state.mip});
      }
      
      src_stages |= src_state.stages;
      dst_stages |= state.dst.stages;

      VkImageMemoryBarrier img_barrier {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        src_state.access,
        state.dst.access,
        src_state.layout,
        state.dst.layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image.vk_image.get_image(),
        {desc.aspect, state.mip, 1, state.layer, 1}
      };
      image_barriers.push_back(img_barrier);
    }

    for (const auto &state : barrier.buffer_barriers) {
      auto &buff = resources.buffers.at(resources.buffer_remap.at(state.buffer_hash));

      auto src_state = state.src;
      if (state.acquire_barrier) {
        src_state = buff.input_state;
      }

      src_stages |= src_state.stages;
      dst_stages |= state.dst.stages;

      VkMemoryBarrier mem_barrier {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        nullptr,
        src_state.access,
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

  gpu::Buffer &RenderResources::get_buffer(std::size_t id) {
    auto index = resources.buffer_remap.at(id);
    return resources.buffers.at(index).vk_buffer;
  }
  
  gpu::Image &RenderResources::get_image(std::size_t id) {
    auto index = resources.image_remap.at(id);
    return resources.images.at(index).vk_image;
  }
  
  VkImageView RenderResources::get_view(const ImageRef &ref) {
    return get_image(ref.get_hash()).get_view(ref.get_range());
  }

}
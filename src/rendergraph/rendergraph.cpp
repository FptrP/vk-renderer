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

  
  void RenderGraph::submit() {
    tracking_state.flush();
    tracking_state.dump_barriers();

    for (auto &task : tasks) {
      task->write_commands();
    }
  }

}
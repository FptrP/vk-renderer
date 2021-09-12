#ifndef BACKBUFFER_SUBPASS_HPP_INCLUDED
#define BACKBUFFER_SUBPASS_HPP_INCLUDED

#include <SDL2/SDL.h>
#include <cmath>

#include "gpu/driver.hpp"
#include "gpu/imgui_context.hpp"
#include "gpu/descriptors.hpp"
#include "gpu_context.hpp"
#include "frame_resources.hpp"
#include "framegraph/framegraph.hpp"


struct UniformData {
  glm::mat4 mvp;
  float r, g, b, a;
};

struct BackbufferSubpass {
  BackbufferSubpass(framegraph::RenderGraph &render_graph, GpuContext &&ctx)
    : graph {render_graph},
      context {ctx},
      renderpass {ctx.device.api_device(), {ctx.swapchain.get_image_info().format}},
      backbuffers {ctx.device.get_swapchain_images(ctx.swapchain)},
      framebuffers {create_framebuffers(ctx, renderpass, backbuffers)},
      imgui_ctx{ctx.window, ctx.instance, ctx.device, (uint32_t)framebuffers.size(), renderpass},
      pipeline {init_pipeline(ctx.device.api_device(), renderpass)},
      ubo {ctx.device.create_dynbuffer<UniformData>(framebuffers.size())}
      {}
  
  void init_graph(uint32_t backbuffer_id, gpu::DescriptorPool &pool) {
    backbuffer_image_id = backbuffer_id;
    framegraph::SubpassDescriptor subpass {graph, "BackbufferSubpass"};
    subpass.use_color_attachment(backbuffer_image_id);
    backbuffer_subpass_id = subpass.flush_task();

    framegraph::Task prepare_present;
    prepare_present.name = "PreparePresent";
    prepare_present.used_images.push_back({
      backbuffer_image_id, 0, 0,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    });

    present_prepare_id = graph.add_task(std::move(prepare_present));

    set_graph_cb(pool);
  }

  void create_fonts(VkCommandBuffer cmd) {
    imgui_ctx.create_fonts(context.device, cmd);
  }

  void set_graph_cb(gpu::DescriptorPool &pool) {

    auto swapchain_info = context.swapchain.get_image_info();

    graph.set_callback(backbuffer_subpass_id, [&, swapchain_info](VkCommandBuffer cmd){
      auto set = pool.allocate_set(pipeline.get_descriptor_set_layout(0));
      
      gpu::DescriptorWriter writer {set};
      writer.bind_dynbuffer(0, ubo);
      writer.write(context.device.api_device());
      

      VkRenderPassBeginInfo renderpass_begin {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = renderpass.api_renderpass(),
        .framebuffer = framebuffers[params.image_id].api_framebuffer(),
        .renderArea = {{0, 0}, swapchain_info.extent2D()},
        .clearValueCount = 0,
        .pClearValues = nullptr
      };

      vkCmdBeginRenderPass(cmd, &renderpass_begin, VK_SUBPASS_CONTENTS_INLINE);
    
      VkClearAttachment clear {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .colorAttachment = 0,
        .clearValue {.color {0.f, 1.f, 0.f}}
      };

      VkClearRect clear_rect {
        .rect {{0, 0}, swapchain_info.extent2D()},
        .baseArrayLayer = 0,
        .layerCount = 1
      };

      VkViewport vp {0.f, 0.f, (float)swapchain_info.width, (float)swapchain_info.height, 0.f, 1.f};

      vkCmdClearAttachments(cmd, 1, &clear, 1, &clear_rect);
      pipeline.bind(cmd);
      gpu::bind_descriptors(cmd, pipeline, 0, {set}, {(uint32_t)ubo.get_offset(params.image_id)});
      vkCmdSetViewport(cmd, 0, 1, &vp);
      vkCmdSetScissor(cmd, 0, 1, &clear_rect.rect);
      vkCmdDraw(cmd, 3, 1, 0, 0);

      imgui_ctx.render(cmd);
      vkCmdEndRenderPass(cmd);
    });
  }

  void new_frame() {
    imgui_ctx.new_frame();
  }

  void process_event(const SDL_Event &event) {
    imgui_ctx.process_event(event);
  }

  void update_graph(FrameGlobal &fg) {
    params.frame_id = fg.frame_index;
    params.image_id = fg.backbuffer_index;
    float time = SDL_GetTicks()/1000.f;

    *ubo.get_mapped_ptr(params.image_id) = UniformData {fg.projection * fg.camera.get_view_mat(), std::abs(std::sin(time)), 0.f, 0.f, 0.f};

  }

private:
  framegraph::RenderGraph &graph;
  GpuContext context;
  gpu::RenderSubpass renderpass;
  std::vector<gpu::Image> backbuffers;
  std::vector<gpu::Framebuffer> framebuffers;
  gpu::ImguiContext imgui_ctx;
  gpu::Pipeline pipeline;
  gpu::DynBuffer<UniformData> ubo;

  uint32_t backbuffer_image_id = 0;
  uint32_t backbuffer_subpass_id = 0;
  uint32_t present_prepare_id = 0; 

  struct CbParams {
    uint32_t frame_id;
    uint32_t image_id;
  } params;

  static std::vector<gpu::Framebuffer> create_framebuffers(GpuContext &ctx, const gpu::RenderSubpass &subpass, std::vector<gpu::Image> &backbuffers) {
    auto ext = ctx.swapchain.get_image_info().extent3D();
    ext.depth = 1;

    std::vector<gpu::Framebuffer> result;
    for (auto &img : backbuffers) {
      gpu::ImageViewRange view {VK_IMAGE_VIEW_TYPE_2D, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      result.push_back(gpu::Framebuffer {ctx.device.api_device(), subpass, ext, {img.get_view(view)}});
    }
    return result;
  }

  static gpu::Pipeline init_pipeline(VkDevice device, const gpu::RenderSubpass &subpass) {
    gpu::ShaderModule vertex {device, "src/shaders/triangle/vert.spv", VK_SHADER_STAGE_VERTEX_BIT};
    gpu::ShaderModule fragment {device, "src/shaders/triangle/frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT};

    gpu::GraphicsPipelineDescriptor registers {1};

    gpu::Pipeline pipeline {device};
    pipeline.init_gfx(subpass, vertex, fragment, registers);
    return pipeline;
  }
};

#endif
#ifndef GBUFFER_SUBPASS_HPP_INCLUDED
#define GBUFFER_SUBPASS_HPP_INCLUDED

#include "framegraph/framegraph.hpp"
#include "gpu/driver.hpp"
#include "gpu/descriptors.hpp"
#include "frame_resources.hpp"

struct GbufConst {
  glm::mat4 mvp;
};

struct GbufferSubpass {
  GbufferSubpass(GpuContext &&ctx, FrameGlobal &state)
    : subpass {
        ctx.device.api_device(),
        {
          state.gbuffer.albedo.get_fmt(),
          state.gbuffer.normal.get_fmt(),
          state.gbuffer.material.get_fmt()
        },
        state.gbuffer.depth.get_fmt()
      },
      framebuffer {
        ctx.device.api_device(),
        subpass,
        state.gbuffer.get_ext_layers(),
        {
          state.gbuffer.albedo.get_view({VK_IMAGE_VIEW_TYPE_2D, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}),
          state.gbuffer.normal.get_view({VK_IMAGE_VIEW_TYPE_2D, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}),
          state.gbuffer.material.get_view({VK_IMAGE_VIEW_TYPE_2D, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}),
          state.gbuffer.depth.get_view({VK_IMAGE_VIEW_TYPE_2D, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}})
        }
      },
      default_pipeline {build_pipeline(ctx, subpass)},
      cbuffer {ctx.device.create_dynbuffer<GbufConst>(state.frames_in_flight)}
  {

  }

  void init_graph(const FrameResources &desc, FrameGlobal &state, framegraph::RenderGraph &graph, gpu::Device &device, gpu::DescriptorPool &pool) {
    framegraph::SubpassDescriptor subpass_desc {graph, "GbufferPass"};
    
    subpass_id = subpass_desc
      .use_color_attachment(desc.gbuf_albedo)
      .use_color_attachment(desc.gbuf_normal)
      .use_color_attachment(desc.gbuf_material)
      .use_depth_attachment(desc.gbuf_depth)
      .flush_task();
    
    graph.set_callback(subpass_id, [&](VkCommandBuffer cmd) {
      auto set = pool.allocate_set(default_pipeline.get_descriptor_set_layout(0));
      gpu::DescriptorWriter updater {set};
      updater.bind_dynbuffer(0, cbuffer);
      updater.write(device.api_device());
      
      *cbuffer.get_mapped_ptr(state.frame_index) = GbufConst {
        state.view_proj
      };

      VkRenderPassBeginInfo render_begin {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = subpass.api_renderpass(),
        .framebuffer = framebuffer.api_framebuffer(),
        .renderArea = VkRect2D {{0, 0}, state.gbuffer.get_extent()},
        .clearValueCount = 0,
        .pClearValues = nullptr
      };

      vkCmdBeginRenderPass(cmd, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
      
      VkClearAttachment clear[] {
        {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .colorAttachment = 0,
          .clearValue {.color {0.f, 0.f, 0.f}}
        },
        {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .colorAttachment = 1,
          .clearValue {.color {0.f, 0.f, 0.f}}
        },
        {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .colorAttachment = 2,
          .clearValue {.color {0.f, 0.f, 0.f}}
        },
        {
          .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
          .colorAttachment = 3,
          .clearValue {.depthStencil {1.f, 0}}
        }
      };

      VkClearRect clear_rect {
        .rect {{0, 0}, state.gbuffer.get_extent()},
        .baseArrayLayer = 0,
        .layerCount = 1
      };

      auto rects = {clear_rect, clear_rect, clear_rect, clear_rect};
      vkCmdClearAttachments(cmd, 4, clear, 4, rects.begin());
      default_pipeline.bind(cmd);
      gpu::bind_descriptors(cmd, default_pipeline, 0, {set}, {(uint32_t)cbuffer.get_offset(state.frame_index)});
      VkViewport vp {0.f, 0.f, (float)clear_rect.rect.extent.width, (float)clear_rect.rect.extent.height, 0.f, 1.f};
      vkCmdSetViewport(cmd, 0, 1, &vp);
      vkCmdSetScissor(cmd, 0, 1, &clear_rect.rect);
      
      auto vbuf = state.scene.get_vertex_buffer().get_api_buffer();
      auto ibuf = state.scene.get_index_buffer().get_api_buffer();
      const auto &mesh = state.scene.get_meshes().at(0);
      uint64_t voffset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &voffset);
      vkCmdBindIndexBuffer(cmd, ibuf, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, mesh.index_count, 1, mesh.index_offset, mesh.vertex_offset, 0);
      vkCmdEndRenderPass(cmd);
    });

  }

private:
  uint32_t subpass_id;
  gpu::RenderSubpass subpass;
  gpu::Framebuffer framebuffer;
  gpu::Pipeline default_pipeline;
  gpu::DynBuffer<GbufConst> cbuffer;

  static gpu::Pipeline build_pipeline(GpuContext &ctx, const gpu::RenderSubpass &subpass) {
    gpu::ShaderModule vertex {ctx.device.api_device(), "src/shaders/gbuf/default_vert.spv", VK_SHADER_STAGE_VERTEX_BIT};
    gpu::ShaderModule fragment {ctx.device.api_device(), "src/shaders/gbuf/default_frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT};

    gpu::GraphicsPipelineDescriptor registers {3};

    registers.set_vertex_input({{0, sizeof(scene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}}, {
      {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, pos)
      },
      {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, norm)
      },
      {
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(scene::Vertex, uv)
      },
    });

    registers.depth_stencil.depthTestEnable = VK_TRUE;
    registers.depth_stencil.depthWriteEnable = VK_TRUE;

    gpu::Pipeline pipeline {ctx.device.api_device()};
    pipeline.init_gfx(subpass, vertex, fragment, registers);
    return pipeline;
  }

};

#endif
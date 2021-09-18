#include "backbuffer_subpass2.hpp"

#include <memory>
#include <vector>

struct ShaderData {
  glm::mat4 mvp;
  glm::vec4 color;
};

struct SubpassData {
  gpu::GraphicsPipeline pipeline;
  std::vector<std::unique_ptr<gpu::Framebuffer>> framebuffers;

  std::unique_ptr<gpu::DynBuffer<ShaderData>> ubo;

  rendergraph::ImageRef backbuff_view;
  bool init = false;
};

struct Nil {};

static gpu::Pipeline init_pipeline(VkDevice device, const gpu::RenderSubpass &subpass) {
  gpu::ShaderModule vertex {device, "src/shaders/triangle/vert.spv", VK_SHADER_STAGE_VERTEX_BIT};
  gpu::ShaderModule fragment {device, "src/shaders/triangle/frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT};

  gpu::GraphicsPipelineDescriptor registers {1};

  gpu::Pipeline pipeline {device};
  pipeline.init_gfx(subpass, vertex, fragment, registers);
  return pipeline;
}

void add_backbuffer_subpass(rendergraph::RenderGraph &graph, gpu::PipelinePool &ppol, glm::mat4 &mvp) {
  graph.add_task<SubpassData>("BackbufSubpass",
    [&](SubpassData &data, rendergraph::RenderGraphBuilder &builder){
      data.backbuff_view = builder.use_backbuffer_attachment();
      data.ubo.reset(new gpu::DynBuffer<ShaderData> {builder.get_gpu().create_dynbuffer<ShaderData>(builder.get_backbuffers_count())});
      data.pipeline.attach(ppol);
    },
    [=, &mvp](SubpassData &data, rendergraph::RenderResources &resources, VkCommandBuffer cmd){
      
      const auto &desc = resources.get_image(data.backbuff_view.get_hash()).get_info();
      auto &dev = resources.get_gpu();

    
      if (!data.init) {
        data.pipeline.set_program("triangle");
        data.pipeline.set_registers({});
        data.pipeline.set_vertex_input({});

        data.framebuffers.resize(resources.get_backbuffers_count());
        data.init = true;
      }
      data.pipeline.set_rendersubpass({false, {desc.format}});

      auto backbuf_id = resources.get_backbuffer_index();

      if (!data.framebuffers[backbuf_id]) {
        data.framebuffers[backbuf_id].reset(new gpu::Framebuffer {dev.api_device(), data.pipeline.get_renderpass(), {desc.width, desc.height, 1}, {resources.get_view(data.backbuff_view)}});
      }

      *data.ubo->get_mapped_ptr(backbuf_id) = ShaderData {mvp, glm::vec4{1, 0, 0, 0}};
      
      auto set = resources.allocate_set(data.pipeline.get_layout(0));
      gpu::DescriptorWriter writer {set};
      writer.bind_dynbuffer(0, *data.ubo);
      writer.write(resources.get_gpu().api_device());

      VkRenderPassBeginInfo renderpass_begin {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = data.pipeline.get_renderpass(),
        .framebuffer = data.framebuffers[backbuf_id]->api_framebuffer(),
        .renderArea = {{0, 0}, desc.extent2D()},
        .clearValueCount = 0,
        .pClearValues = nullptr
      };

      VkClearAttachment clear {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .colorAttachment = 0,
        .clearValue {.color {0.f, 0.f, 0.f}}
      };

      VkClearRect clear_rect {
        .rect {{0, 0}, desc.extent2D()},
        .baseArrayLayer = 0,
        .layerCount = 1
      };

      VkRect2D scissors {{0, 0}, desc.extent2D()};
      VkViewport viewport {0.f, 0.f, (float)desc.width, (float)desc.height, 0.f, 1.f};

      vkCmdBeginRenderPass(cmd, &renderpass_begin, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdClearAttachments(cmd, 1, &clear, 1, &clear_rect);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, data.pipeline.get_pipeline());
      uint32_t offs = data.ubo->get_offset(backbuf_id);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, data.pipeline.get_pipeline_layout(), 0, 1, &set, 1, &offs);
      vkCmdSetViewport(cmd, 0, 1, &viewport);
      vkCmdSetScissor(cmd, 0, 1, &scissors);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);

    });

  graph.add_task<Nil>("presentPrepare",
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.prepare_backbuffer();
  },
  [=](Nil &, rendergraph::RenderResources&, VkCommandBuffer){

  });
}
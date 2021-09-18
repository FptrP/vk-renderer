#include "backbuffer_subpass2.hpp"

#include <memory>
#include <vector>

struct ShaderData {
  glm::mat4 mvp;
  glm::vec4 color;
};

struct SubpassData {
  gpu::GraphicsPipeline pipeline;
  rendergraph::ImageViewId backbuff_view;
  bool init = false;
};

struct Nil {};

std::unique_ptr<gpu::DynBuffer<ShaderData>> ubo;

void add_backbuffer_subpass(rendergraph::RenderGraph &graph, gpu::PipelinePool &ppol, glm::mat4 &mvp) {
  graph.add_task<SubpassData>("BackbufSubpass",
    [&](SubpassData &data, rendergraph::RenderGraphBuilder &builder){
      data.backbuff_view = builder.use_backbuffer_attachment();
      
      data.pipeline.attach(ppol);
    },
    [=, &mvp](SubpassData &data, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      if (!ubo) {
        ubo.reset(new gpu::DynBuffer<ShaderData> {resources.get_gpu().create_dynbuffer<ShaderData>(resources.get_backbuffers_count())});
      }
      
      const auto &desc = resources.get_image(data.backbuff_view).get_info();


      gpu::Registers regs {};
      data.pipeline.set_program("triangle");
      data.pipeline.set_registers(regs);
      data.pipeline.set_vertex_input({});
      
      data.pipeline.set_rendersubpass({false, {desc.format}});

      auto backbuf_id = resources.get_backbuffer_index();

      *ubo->get_mapped_ptr(backbuf_id) = ShaderData {mvp, glm::vec4{1, 0, 0, 0}};
      
      auto set = resources.allocate_set(data.pipeline.get_layout(0));
      gpu::DescriptorWriter writer {set};
      writer.bind_dynbuffer(0, *ubo);
      writer.write(resources.get_gpu().api_device());


      VkRect2D scissors {{0, 0}, desc.extent2D()};
      VkViewport viewport {0.f, 0.f, (float)desc.width, (float)desc.height, 0.f, 1.f};
      uint32_t offs = ubo->get_offset(backbuf_id);

      cmd.set_framebuffer(desc.width, desc.height, {resources.get_view(data.backbuff_view)});
      cmd.bind_pipeline(data.pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.bind_descriptors_graphics(0, {set}, {offs});
      cmd.bind_viewport(viewport);
      cmd.bind_scissors(scissors);
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });

  graph.add_task<Nil>("presentPrepare",
  [&](Nil &, rendergraph::RenderGraphBuilder &builder){
    builder.prepare_backbuffer();
  },
  [=](Nil &, rendergraph::RenderResources&, gpu::CmdContext &){

  });
}
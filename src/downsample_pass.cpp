#include "downsample_pass.hpp"
#include "gpu/gpu.hpp"

void downsample_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth_tex) {
  auto &desc = graph.get_descriptor(depth_tex);
  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;

  auto pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("downsample_depth");
  pipeline.set_registers(regs);
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({true, {desc.format}});

  struct Input {
    rendergraph::ImageViewId depth_tex;
    rendergraph::ImageViewId depth_rt;
  };

  for (uint32_t i = 1; i < desc.mip_levels; i++) {
    graph.add_task<Input>("DownsampleDepth",
      [&](Input &input, rendergraph::RenderGraphBuilder &builder){
        input.depth_rt = builder.use_depth_attachment(depth_tex, i, 0);
        input.depth_tex = builder.sample_image(depth_tex, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, i - 1, 1, 0, 1);
      },
      [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
        auto set = resources.allocate_set(pipeline, 0); 
        gpu::write_set(set, 
          gpu::TextureBinding {0, resources.get_view(input.depth_tex), sampler});

        uint32_t w = desc.width/(1 << i), h = desc.height/(1 << i);

        cmd.set_framebuffer(w, h, {resources.get_view(input.depth_rt)});
        cmd.bind_pipeline(pipeline);
        cmd.bind_descriptors_graphics(0, {set});
        cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
        cmd.bind_scissors(0, 0, w, h);
        cmd.draw(3, 1, 0, 0);
        cmd.end_renderpass();
      });
  }

  

}
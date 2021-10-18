#include "gtao.hpp"

rendergraph::ImageResourceId create_gtao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo info {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  return graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
}

void add_gtao_main_pass(
  rendergraph::RenderGraph &graph,
  const GTAOParams &params,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal,
  rendergraph::ImageResourceId out_image)
{
  auto pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("gtao_main");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({false, {graph.get_descriptor(out_image).format}});
  
  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
  
  struct PassData {
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
  };

  graph.add_task<PassData>("GTAO",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.rt = builder.use_color_attachment(out_image, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler});
      
      const auto &image_info = resources.get_image(input.rt).get_info();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_view(input.rt)});
      cmd.bind_pipeline(pipeline);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {block.offset});
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });
}
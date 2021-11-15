#include "gtao.hpp"
#include <cstdlib>

rendergraph::ImageResourceId create_gtao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo info {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  return graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
}

GTAO::GTAO(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo info {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  auto usage = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;

  raw = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  filtered = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  prev_frame = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  output = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  
  main_pipeline = gpu::create_compute_pipeline();
  main_pipeline.set_program("gtao_compute_main");

  filter_pipeline = gpu::create_compute_pipeline();
  filter_pipeline.set_program("gtao_filter");

  reproject_pipeline = gpu::create_compute_pipeline();
  reproject_pipeline.set_program("gtao_reproject");

  main_pipeline_gfx = gpu::create_graphics_pipeline();
  main_pipeline_gfx.set_program("gtao_main");
  main_pipeline_gfx.set_registers({});
  main_pipeline_gfx.set_vertex_input({});
  main_pipeline_gfx.set_rendersubpass({false, {graph.get_descriptor(raw).format}});

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}

void GTAO::add_main_pass(
  rendergraph::RenderGraph &graph,
  const GTAOParams &params,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal)
{

  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
  };

  graph.add_task<PassData>("GTAO_main",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(main_pipeline, 0);
    
      gpu::write_set(set,
        gpu::UBOBinding {0, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler},
        gpu::StorageTextureBinding {3, resources.get_view(input.out)}
        );

      const auto &extent = resources.get_image(input.out).get_extent();

      cmd.bind_pipeline(main_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {block.offset});
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });

}

void GTAO::add_filter_pass(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth)
{

  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId raw_gtao;
  };
  
  struct FilterData {
    float znear;
    float zfar;
  };

  FilterData filter_params {params.znear, params.zfar};

  graph.add_task<PassData>("GTAO_filter",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.raw_gtao = builder.sample_image(raw, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(filter_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.raw_gtao), sampler},
        gpu::StorageTextureBinding {2, resources.get_view(input.out)}
      );

      const auto &extent = resources.get_image(input.out).get_extent();

      cmd.bind_pipeline(filter_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {});
      cmd.push_constants_compute(0, sizeof(filter_params), &filter_params);
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}

void GTAO::add_reprojection_pass(
    rendergraph::RenderGraph &graph,
    const GTAOReprojection &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId prev_depth)
{
  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId gtao;
    rendergraph::ImageViewId prev_gtao;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId prev_depth;
  };

  graph.add_task<PassData>("GTAO_reproject",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.prev_depth = builder.sample_image(prev_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.gtao = builder.sample_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT);
      input.prev_gtao = builder.sample_image(prev_frame, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(output, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOReprojection>();
      *block.ptr = params;

      auto set = resources.allocate_set(reproject_pipeline, 0);
    
      gpu::write_set(set,
        gpu::UBOBinding {0, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.prev_depth), sampler},
        gpu::TextureBinding {3, resources.get_view(input.gtao), sampler},
        gpu::TextureBinding {4, resources.get_view(input.prev_gtao), sampler},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)}
      );

      const auto &extent = resources.get_image(input.out).get_extent();

      cmd.bind_pipeline(reproject_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {block.offset});
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}

void GTAO::add_main_pass_graphics(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal)
{
  struct PushConstants {
    float angle_offset;
  };

  struct PassData {
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
  };

  const float angle_offsets[6] {60.f, 300.f, 180.f, 240.f, 120.f, 0.f};
  float base_angle = angle_offsets[frame_count % (sizeof(angle_offsets)/sizeof(float))];
  base_angle += rand()/float(RAND_MAX) - 0.5;

  PushConstants constants {base_angle};

  frame_count += 1;

  graph.add_task<PassData>("GTAO",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.rt = builder.use_color_attachment(raw, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GTAOParams>();
      *block.ptr = params;

      auto set = resources.allocate_set(main_pipeline_gfx, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), block},
        gpu::TextureBinding {2, resources.get_view(input.norm), sampler});
      
      const auto &image_info = resources.get_image(input.rt).get_info();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_view(input.rt)});
      cmd.bind_pipeline(main_pipeline_gfx);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {block.offset});
      cmd.push_constants_graphics(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants), &constants);
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    });
}
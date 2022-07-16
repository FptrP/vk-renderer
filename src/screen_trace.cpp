#include "screen_trace.hpp"

ScreenSpaceTrace::ScreenSpaceTrace(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo info {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  auto usage = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;

  raw = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  filtered = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);
  accumulated  = graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, usage);

  trace_pipeline = gpu::create_compute_pipeline();
  trace_pipeline.set_program("screen_trace_main");

  filter_pipeline = gpu::create_compute_pipeline();
  filter_pipeline.set_program("screen_trace_filter");

  accum_pipeline = gpu::create_compute_pipeline();
  accum_pipeline.set_program("screen_trace_accumulate");

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}

void ScreenSpaceTrace::add_main_pass(
  rendergraph::RenderGraph &graph,
  const ScreenTraceParams &params,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal,
  rendergraph::ImageResourceId color,
  rendergraph::ImageResourceId material)
{
  struct GpuParams {
    glm::mat4 camera_normal;
    float random_offset;
    float angle_offset;
    float fovy;
    float aspect;
    float znear;
    float zfar;
  };
  
  struct PassData {
    rendergraph::ImageViewId out;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId norm;
    rendergraph::ImageViewId color;
    rendergraph::ImageViewId material;
  };

  const float angle_offsets[] {60.f, 300.f, 180.f, 240.f, 120.f, 0.f, 300.f, 60.f, 180.f, 120.f, 240.f, 0.f};
  float base_angle = angle_offsets[frame_count % (sizeof(angle_offsets)/sizeof(float))]/360.f;
  base_angle += random_floats(generator) - 0.5;
  frame_count += 1;

  float random_offset = random_floats(generator);

  graph.add_task<PassData>("ScreenTrace",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.norm = builder.sample_image(normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.color = builder.sample_image(color, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(raw, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto block = cmd.allocate_ubo<GpuParams>();
      block.ptr->camera_normal = params.normal_mat;
      block.ptr->angle_offset = base_angle;
      block.ptr->aspect = params.aspect;
      block.ptr->fovy = params.fovy;
      block.ptr->znear = params.znear;
      block.ptr->random_offset = random_offset;
      block.ptr->zfar = params.zfar;

      auto set = resources.allocate_set(trace_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.norm), sampler},
        gpu::TextureBinding {2, resources.get_view(input.color), sampler},
        gpu::TextureBinding {3, resources.get_view(input.material), sampler},
        gpu::StorageTextureBinding {4, resources.get_view(input.out)},
        gpu::UBOBinding {5, cmd.get_ubo_pool(), block}
      );

      const auto &extent = resources.get_image(input.out)->get_extent();

      cmd.bind_pipeline(trace_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {block.offset});
      cmd.dispatch(extent.width/8, extent.height/8, 1);
    }); 
}

void ScreenSpaceTrace::add_filter_pass(
  rendergraph::RenderGraph &graph,
  const ScreenTraceParams &params,
  rendergraph::ImageResourceId depth)
{
  struct GpuParams {
    float znear;
    float zfar;
  };
  
  struct PassData {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId raw;
    rendergraph::ImageViewId filtered;
  };

  GpuParams push_constants {params.znear, params.zfar};

  graph.add_task<PassData>("ScreenTraceFilter",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.raw = builder.sample_image(raw, VK_SHADER_STAGE_COMPUTE_BIT);
      input.filtered = builder.use_storage_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(filter_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.raw), sampler},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::StorageTextureBinding {2, resources.get_view(input.filtered)}
      );

      const auto &extent = resources.get_image(input.filtered)->get_extent();

      cmd.bind_pipeline(filter_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {});
      cmd.push_constants_compute(0, sizeof(push_constants), &push_constants);
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}

void ScreenSpaceTrace::add_accumulate_pass(
  rendergraph::RenderGraph &graph,
  const ScreenTraceParams &params,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId prev_depth)
{
  struct GpuParams {
    float fovy;
    float aspect;
    float znear;
    float zfar;
  };
  
  struct PassData {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId prev_depth;
    rendergraph::ImageViewId filtered;
    rendergraph::ImageViewId accum;
  };

  GpuParams push_constants {params.fovy, params.aspect, params.znear, params.zfar};

  graph.add_task<PassData>("ScreenTraceAccumulate",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.prev_depth = builder.sample_image(prev_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.filtered = builder.sample_image(filtered, VK_SHADER_STAGE_COMPUTE_BIT);
      input.accum = builder.use_storage_image(accumulated, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(accum_pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.prev_depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.filtered), sampler},
        gpu::StorageTextureBinding {3, resources.get_view(input.accum)}
      );

      const auto &extent = resources.get_image(input.accum)->get_extent();

      cmd.bind_pipeline(accum_pipeline);
      cmd.bind_descriptors_compute(0, {set}, {});
      cmd.push_constants_compute(0, sizeof(push_constants), &push_constants);
      cmd.dispatch(extent.width/8, extent.height/4, 1);
    });
}
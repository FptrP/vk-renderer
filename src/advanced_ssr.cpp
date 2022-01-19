#include "advanced_ssr.hpp"
#include "imgui_pass.hpp"

#define HALTON_SEQ_SIZE 128

static float halton_elem(uint32_t index, uint32_t base) {
  float f = 1;
  float r = 0;
  uint32_t current = index;
  do
  {
    f = f / base;
    r = r + f * (current % base);
    current = glm::floor(current / float(base));
  } while (current > 0);
  
  return r;
}

std::vector<glm::vec4> halton23_seq(uint32_t count) {
  std::vector<glm::vec4> seq {};
  seq.reserve(count);

  for (uint32_t iter = 0; iter < count; iter++) {
    glm::vec4 elem {0, 0, 0, 0};
    elem.x = halton_elem(iter + 1, 2);
    elem.y = halton_elem(iter + 1, 3);
    seq.push_back(elem);
  }

  return seq;
}

AdvancedSSR::AdvancedSSR(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h) {
  trace_pass = gpu::create_compute_pipeline();
  trace_pass.set_program("sssr_trace");
  
  filter_pass = gpu::create_compute_pipeline();
  filter_pass.set_program("sssr_filter");

  auto halton_samples = halton23_seq(HALTON_SEQ_SIZE);
  const uint64_t bytes = sizeof(halton_samples[0]) * HALTON_SEQ_SIZE;

  halton_buffer.create(VMA_MEMORY_USAGE_CPU_TO_GPU, bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  std::memcpy(halton_buffer.get_mapped_ptr(), halton_samples.data(), bytes);
  
  gpu::ImageInfo rays_info {VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w/2, h/2};
  rays = graph.create_image(VK_IMAGE_TYPE_2D, rays_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  gpu::ImageInfo reflections_info {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, w/2, h/2};
  reflections = graph.create_image(VK_IMAGE_TYPE_2D, reflections_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_STORAGE_BIT);

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER); 
}

struct TraceParams {
  glm::mat4 normal_mat;
  uint32_t frame_random;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

void AdvancedSSR::run_trace_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  TraceParams config {
    params.normal_mat,
    counter,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };

  struct PushConstants {
    float max_roughness;  
  };

  PushConstants push_consts {settings.max_rougness};

  if (settings.update_random) {
    counter++;
    counter = counter % 16;
  }

  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId out;
  };
  
  auto mips_count = graph.get_descriptor(gbuff.depth).mip_levels;

  graph.add_task<Input>("SSSR_trace",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 1, mips_count - 1, 0, 1);
      input.normal = builder.sample_image(gbuff.downsampled_normals, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out = builder.use_storage_image(rays, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(trace_pass, 0);
      auto blk = cmd.allocate_ubo<TraceParams>();
      *blk.ptr = config;

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::UBOBinding {3, cmd.get_ubo_pool(), blk},
        gpu::UBOBinding {4, halton_buffer},
        gpu::StorageTextureBinding {5, resources.get_view(input.out)});
      
      auto ext = resources.get_image(input.out).get_extent();
      cmd.bind_pipeline(trace_pass);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset, 0});
      cmd.push_constants_compute(0, sizeof(push_consts), &push_consts);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 7)/8, 1);
    });
}

#define NORMALIZE_REFLECTIONS 1
#define ACCUMULATE_REFLECTIONS 2
#define BILATERAL_FILTER 4

void AdvancedSSR::run_filter_pass(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  TraceParams config {
    params.normal_mat,
    counter,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };

  struct Input {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId rays;
    rendergraph::ImageViewId reflection; 
  };

  struct PushConstants {
    uint32_t render_flags;
  };

  PushConstants pc {0u};
  pc.render_flags |= settings.normalize_reflections? NORMALIZE_REFLECTIONS : 0;
  pc.render_flags |= settings.accumulate_reflections? ACCUMULATE_REFLECTIONS : 0;
  pc.render_flags |= settings.bilateral_filter? BILATERAL_FILTER : 0;

  graph.add_task<Input>("SSSR_filter",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuff.depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 10, 0, 1);
      input.normal = builder.sample_image(gbuff.normal, VK_SHADER_STAGE_COMPUTE_BIT);
      input.albedo = builder.sample_image(gbuff.albedo, VK_SHADER_STAGE_COMPUTE_BIT);
      input.rays = builder.sample_image(rays, VK_SHADER_STAGE_COMPUTE_BIT);
      input.material = builder.sample_image(gbuff.material, VK_SHADER_STAGE_COMPUTE_BIT);
      input.reflection = builder.use_storage_image(reflections, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      auto set = resources.allocate_set(filter_pass, 0);
      auto blk = cmd.allocate_ubo<TraceParams>();
      *blk.ptr = config;

      gpu::write_set(set, 
        gpu::TextureBinding {0, resources.get_view(input.rays), sampler},
        gpu::TextureBinding {1, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {2, resources.get_view(input.albedo), sampler},
        gpu::TextureBinding {3, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {4, resources.get_view(input.material), sampler},
        gpu::StorageTextureBinding {5, resources.get_view(input.reflection)},
        gpu::UBOBinding {6, cmd.get_ubo_pool(), blk});
      
      auto ext = resources.get_image(input.reflection).get_extent();
      cmd.bind_pipeline(filter_pass);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      cmd.push_constants_compute(0, sizeof(pc), &pc);
      cmd.dispatch((ext.width + 7)/8, (ext.height + 7)/8, 1);
    });
}

void AdvancedSSR::run(
  rendergraph::RenderGraph &graph,
  const AdvancedSSRParams &params,
  const Gbuffer &gbuff)
{
  run_trace_pass(graph, params, gbuff);
  run_filter_pass(graph, params, gbuff);
}

void AdvancedSSR::render_ui() {
  ImGui::Begin("SSSR");
  ImGui::SliderFloat("Max Roughness", &settings.max_rougness, 0.f, 1.f);
  ImGui::Checkbox("Enable normalization", &settings.normalize_reflections);
  ImGui::Checkbox("Enable accumulation", &settings.accumulate_reflections);
  ImGui::Checkbox("Enable random rays", &settings.update_random);
  ImGui::Checkbox("Enable bilateral filter", &settings.bilateral_filter);
  ImGui::End();
}
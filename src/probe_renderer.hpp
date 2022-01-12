#ifndef PROBE_RENDERER_HPP_INCLUDED
#define PROBE_RENDERER_HPP_INCLUDED

#include "scene_renderer.hpp"

const uint32_t PROBE_SIZE = 512;
const uint32_t CUBE_SIZE = 256;

struct OctahedralProbe {
  OctahedralProbe(rendergraph::RenderGraph &graph, uint32_t size = PROBE_SIZE);
  
  glm::vec3 pos {0.f, 0.f, 0.f};
  rendergraph::ImageResourceId color;
  rendergraph::ImageResourceId depth;
};

struct OctahedralProbeGrid {
  OctahedralProbeGrid(rendergraph::RenderGraph &graph, uint32_t grid_sz = 4, uint32_t size = PROBE_SIZE);
  
  glm::vec3 min {0.f, 0.f, 0.f};
  glm::vec3 max {0.f, 0.f, 0.f};
  uint32_t grid_size {0};

  rendergraph::ImageResourceId color_array;
  rendergraph::ImageResourceId depth_array;
};

struct ProbeRenderer {
  ProbeRenderer(rendergraph::RenderGraph &graph, uint32_t cubemap_res = CUBE_SIZE);

  void render_cubemap(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, const glm::vec3 pos);
  void render_probe(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, const glm::vec3 pos, OctahedralProbe &probe);
  void render_probe_grid(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, glm::vec3 min, glm::vec3 max, OctahedralProbeGrid &probe_grid);
  
private:
  rendergraph::ImageResourceId cubemap_color;
  rendergraph::ImageResourceId cubemap_distance;
  rendergraph::ImageResourceId rt_depth;

  gpu::GraphicsPipeline cubemap_pass;
  gpu::ComputePipeline octprobe_pass;
  gpu::GraphicsPipeline downsample_pass;

  VkSampler sampler;

  void render_side(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, uint32_t side, glm::mat4 view);
  void render_octahedral(rendergraph::RenderGraph &graph, OctahedralProbe &probe);
  void probe_downsample(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId probe_depth);
};

struct ProbeTraceParams {
  glm::mat4 inv_view; //camera -> world
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct ProbeTracePass {
  ProbeTracePass();

  void run(rendergraph::RenderGraph &graph, OctahedralProbe &probe, rendergraph::ImageResourceId gbuffer_depth, rendergraph::ImageResourceId gbuffer_norm, rendergraph::ImageResourceId out_image,  const ProbeTraceParams &params);

private:
  gpu::ComputePipeline trace_pass;
  VkSampler sampler;
};

#endif
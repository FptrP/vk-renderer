#ifndef ADVANCED_SSR_HPP_INCLUDED
#define ADVANCED_SSR_HPP_INCLUDED

#include <rendergraph/rendergraph.hpp>
#include "scene_renderer.hpp"

struct AdvancedSSRParams {
  glm::mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

std::vector<glm::vec4> halton23_seq(uint32_t count);

struct AdvancedSSR {
  void render_ui();

  AdvancedSSR(rendergraph::RenderGraph &graph, uint32_t w, uint32_t h);
  void run(
    rendergraph::RenderGraph &graph,
    const AdvancedSSRParams &params,
    const Gbuffer &gbuff);

  rendergraph::ImageResourceId get_ouput() const { return reflections; }
  rendergraph::ImageResourceId get_rays() const { return rays; }
  rendergraph::ImageResourceId get_blurred() const { return blurred_reflection; }

private:
  gpu::Buffer halton_buffer;

  rendergraph::BufferResourceId reflective_indirect;
  rendergraph::BufferResourceId glossy_indirect;
  rendergraph::BufferResourceId reflective_tiles;
  rendergraph::BufferResourceId glossy_tiles;
  
  gpu::ComputePipeline trace_pass;
  gpu::ComputePipeline filter_pass;
  gpu::ComputePipeline blur_pass;
  gpu::ComputePipeline classification_pass;
  gpu::ComputePipeline trace_indirect_pass;

  VkSampler sampler;

  rendergraph::ImageResourceId rays;
  rendergraph::ImageResourceId reflections;
  rendergraph::ImageResourceId blurred_reflection;

  uint32_t counter {0u};

  struct {
    float max_rougness = 1.f;
    float glossy_roughness_value = 0.5f;
    bool normalize_reflections = true;
    bool accumulate_reflections = true;
    bool bilateral_filter = true;
    bool update_random = true;
    bool use_blur = true;
    int max_accumulated_rays = 16;
  } settings;

  void run_trace_pass(
    rendergraph::RenderGraph &graph,
    const AdvancedSSRParams &params,
    const Gbuffer &gbuff);
  
  void run_trace_indirect_pass(
    rendergraph::RenderGraph &graph,
    const AdvancedSSRParams &params,
    const Gbuffer &gbuff);

  void run_filter_pass(
    rendergraph::RenderGraph &graph,
    const AdvancedSSRParams &params,
    const Gbuffer &gbuff);
  
  void run_blur_pass(
    rendergraph::RenderGraph &graph,
    const AdvancedSSRParams &params,
    const Gbuffer &gbuff);
  
  void run_classification_pass(
    rendergraph::RenderGraph &graph,
    const AdvancedSSRParams &params,
    const Gbuffer &gbuff);

  void clear_indirect_params(rendergraph::RenderGraph &graph);
};

#endif
#ifndef SCREEN_TRACE_HPP_INCLUDED
#define SCREEN_TRACE_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"

struct ScreenTraceParams {
  glm::mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct ScreenSpaceTrace {
  ScreenSpaceTrace(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

  void add_main_pass(
    rendergraph::RenderGraph &graph,
    const ScreenTraceParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal,
    rendergraph::ImageResourceId color,
    rendergraph::ImageResourceId material);

  void add_filter_pass(
    rendergraph::RenderGraph &graph,
    const ScreenTraceParams &params,
    rendergraph::ImageResourceId depth);

  void add_accumulate_pass(
    rendergraph::RenderGraph &graph,
    const ScreenTraceParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId prev_depth);

  rendergraph::ImageResourceId raw;
  rendergraph::ImageResourceId filtered;
  rendergraph::ImageResourceId accumulated;
  //rendergraph::ImageResourceId output;
private:
  

  gpu::ComputePipeline trace_pipeline;
  gpu::ComputePipeline filter_pipeline;
  gpu::ComputePipeline accum_pipeline;
  
  uint32_t frame_count = 0;
  VkSampler sampler;
};


#endif
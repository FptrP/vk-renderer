#ifndef GTAO_HPP_INCLUDED
#define GTAO_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"

rendergraph::ImageResourceId create_gtao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

struct GTAOParams {
  glm::mat4 normal_mat;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct GTAOReprojection {
  glm::mat4 camera_to_prev_frame;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct GTAO {
  GTAO(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

  void add_main_pass(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal);
  
  void add_main_pass_graphics(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId normal);
  
  void add_filter_pass(
    rendergraph::RenderGraph &graph,
    const GTAOParams &params,
    rendergraph::ImageResourceId depth);

  void add_reprojection_pass(
    rendergraph::RenderGraph &graph,
    const GTAOReprojection &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId prev_depth);

  void add_accumulate_pass(
    rendergraph::RenderGraph &graph,
    const GTAOReprojection &params,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId prev_depth);

  rendergraph::ImageResourceId raw; //output of main pass
  rendergraph::ImageResourceId filtered; //output of filter pass
  rendergraph::ImageResourceId prev_frame; //previous frame
  rendergraph::ImageResourceId output; //final
  rendergraph::ImageResourceId accumulated_ao;

private:

  gpu::GraphicsPipeline main_pipeline_gfx;
  gpu::ComputePipeline main_pipeline;
  gpu::ComputePipeline filter_pipeline;
  gpu::ComputePipeline reproject_pipeline;
  gpu::ComputePipeline accumulate_pipeline;

  uint32_t frame_count = 0;

  VkSampler sampler;
};

#endif
#ifndef DEFFERED_SHADING_HPP_INCLUDED
#define DEFFERED_SHADING_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene_renderer.hpp"

struct DeferedShadingPass {

  DeferedShadingPass(rendergraph::RenderGraph &graph);

  void update_params(const glm::mat4 &camera, const glm::mat4 &shadow, float fovy, float aspect, float znear, float zfar);
  void draw(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const rendergraph::ImageResourceId &shadow, const rendergraph::ImageResourceId &ssao, const rendergraph::ImageResourceId &out_image);
  
private:

  gpu::GraphicsPipeline pipeline;
  VkSampler sampler;
  rendergraph::BufferResourceId ubo_consts;
};


#endif
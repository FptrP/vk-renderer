#ifndef SSAO_HPP_INCLUDED
#define SSAO_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"

rendergraph::ImageResourceId create_ssao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

struct SSAOInParams {
  glm::mat4 projection;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

struct SSAOPass {
  SSAOPass(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId target);

  void draw(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth, rendergraph::ImageResourceId target, const SSAOInParams &params);

private:
  gpu::GraphicsPipeline pipeline;
  std::vector<glm::vec3> sphere_samples;
  VkSampler sampler;
};


#endif
#ifndef GBUFFER_SUBPASS2_HPP_INCLUDED
#define GBUFFER_SUBPASS2_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"
#include "scene/scene.hpp"
#include "gpu/gpu.hpp"

struct GbufferShaderData {
  glm::mat4 mvp;
};


struct GbufferData {
  GbufferData(rendergraph::RenderGraph &rendergraph, uint32_t w, uint32_t h);

  rendergraph::ImageResourceId albedo;
  rendergraph::ImageResourceId normal;
  rendergraph::ImageResourceId material;
  rendergraph::ImageResourceId depth;

private:
  gpu::GraphicsPipeline pipeline;
  gpu::DynBuffer<GbufferShaderData> ubo;
  uint32_t width, height;
  friend void add_gbuffer_subpass(GbufferData &gbuf, rendergraph::RenderGraph &rendergraph, scene::CompiledScene &scene, glm::mat4 mvp, rendergraph::ImageResourceId img, VkSampler sampler);
};

void add_gbuffer_subpass(GbufferData &gbuf, rendergraph::RenderGraph &rendergraph, scene::CompiledScene &scene, glm::mat4 mvp, rendergraph::ImageResourceId img, VkSampler sampler);

#endif
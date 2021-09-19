#ifndef GBUFFER_SUBPASS2_HPP_INCLUDED
#define GBUFFER_SUBPASS2_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"
#include "scene/scene.hpp"
#include "gpu/pipelines.hpp"

struct GbufferShaderData {
  glm::mat4 mvp;
};


struct GbufferData {
  GbufferData(gpu::PipelinePool &pipelines, rendergraph::RenderGraph &rendergraph, gpu::Device &device, uint32_t w, uint32_t h);

  rendergraph::ImageResourceId albedo;
  rendergraph::ImageResourceId normal;
  rendergraph::ImageResourceId material;
  rendergraph::ImageResourceId depth;

private:
  gpu::GraphicsPipeline pipeline;
  gpu::DynBuffer<GbufferShaderData> ubo;
  uint32_t width, height;
  friend void add_gbuffer_subpass(GbufferData &gbuf, rendergraph::RenderGraph &rendergraph, scene::Scene &scene, glm::mat4 mvp);
};


void add_gbuffer_subpass(GbufferData &gbuf, rendergraph::RenderGraph &rendergraph, scene::Scene &scene, glm::mat4 mvp);

#endif
#ifndef SCENE_RENDERER_HPP_INCLUDED
#define SCENE_RENDERER_HPP_INCLUDED

#include "scene/scene.hpp"
#include "gpu/gpu.hpp"
#include "rendergraph/rendergraph.hpp"

#include <optional>
#include <memory>

struct Gbuffer {
  Gbuffer(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height);

  rendergraph::ImageResourceId albedo;
  rendergraph::ImageResourceId normal;
  rendergraph::ImageResourceId material;
  rendergraph::ImageResourceId depth;

  uint32_t w, h;
};

struct SceneRenderer {
  SceneRenderer(scene::CompiledScene &s) : target {s} {}

  void init_pipeline(rendergraph::RenderGraph &graph, const Gbuffer &buffer);
  void update_scene(const glm::mat4 &mvp);
  void draw(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer);

  struct DrawCall {
    uint32_t transform;
    uint32_t mesh;
  };
  
private:
  scene::CompiledScene &target;
  gpu::GraphicsPipeline opaque_pipeline;
  std::vector<DrawCall> draw_calls;
  VkSampler sampler;
  
  rendergraph::BufferResourceId transform_buffer;
  rendergraph::BufferResourceId view_proj_buffer;
  void build_scene();

};


#endif
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
  rendergraph::ImageResourceId downsampled_normals;
  rendergraph::ImageResourceId material;
  rendergraph::ImageResourceId depth;
  rendergraph::ImageResourceId prev_depth;

  uint32_t w, h;
};

struct SceneRenderer {
  SceneRenderer(scene::CompiledScene &s) : target {s} {}

  void init_pipeline(rendergraph::RenderGraph &graph, const Gbuffer &buffer);
  void update_scene(const glm::mat4 &camera, const glm::mat4 &projection);
  
  void draw(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer);
  void render_shadow(rendergraph::RenderGraph &graph, const glm::mat4 &shadow_mvp, rendergraph::ImageResourceId out_tex, uint32_t layer);

  struct DrawCall {
    uint32_t transform;
    uint32_t mesh;
  };
  
  const std::vector<DrawCall> &get_drawcalls() const { return draw_calls; }
  const std::vector<VkImageView> &get_images() const { return scene_image_views; }
  
  rendergraph::BufferResourceId get_scene_transforms() const { return transform_buffer; }
  const scene::CompiledScene &get_target() const { return target; }

private:
  scene::CompiledScene &target;
  gpu::GraphicsPipeline opaque_pipeline;
  gpu::GraphicsPipeline shadow_pipeline;

  std::vector<VkImageView> scene_image_views;
  std::vector<DrawCall> draw_calls;
  VkSampler sampler;
  
  rendergraph::BufferResourceId transform_buffer;
  rendergraph::BufferResourceId view_proj_buffer;
};


#endif
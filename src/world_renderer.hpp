#ifndef WORLD_RENDERER_HPP_INCLUDED
#define WORLD_RENDERER_HPP_INCLUDED

#include "gpu/gpu.hpp"
#include "scene/scene.hpp"
#include "scene/scene_as.hpp"
#include "rendergraph/rendergraph.hpp"

struct FrameParams {
  glm::vec4 projection_params;
  glm::mat4 projection;
  glm::mat4 camera_view;
  glm::mat4 inverse_camera_view;
  glm::mat4 normal_matrix;
};

struct WorldRenderer {
  WorldRenderer();

  void init();
  void draw();

private:
  rendergraph::RenderGraph graph;
};

#endif
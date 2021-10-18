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

void add_gtao_main_pass(
  rendergraph::RenderGraph &graph, 
  const GTAOParams &params,
  rendergraph::ImageResourceId depth,
  rendergraph::ImageResourceId normal,
  rendergraph::ImageResourceId out_image);

#endif
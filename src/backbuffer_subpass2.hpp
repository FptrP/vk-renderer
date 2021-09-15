#ifndef BACKBUFF_SUBPASS2_HPP_INCLUDED
#define BACKBUFF_SUBPASS2_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"

void add_backbuffer_subpass(rendergraph::RenderGraph &graph, glm::mat4 &mvp);

#endif
#ifndef BACKBUFF_SUBPASS2_HPP_INCLUDED
#define BACKBUFF_SUBPASS2_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"
#include "gpu/gpu.hpp"

void add_backbuffer_subpass(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId draw_img, VkSampler sampler);
void add_backbuffer_subpass(rendergraph::RenderGraph &graph, gpu::Image &image, VkSampler sampler);
void add_present_subpass(rendergraph::RenderGraph &graph);
#endif
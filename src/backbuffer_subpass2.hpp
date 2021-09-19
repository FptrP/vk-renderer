#ifndef BACKBUFF_SUBPASS2_HPP_INCLUDED
#define BACKBUFF_SUBPASS2_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"
#include "scene/camera.hpp"
#include "gpu/pipelines.hpp"

void add_backbuffer_subpass(rendergraph::ImageResourceId draw_img, gpu::Sampler &sampler, rendergraph::RenderGraph &graph, gpu::PipelinePool &ppol);

#endif
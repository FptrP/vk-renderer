#ifndef DOWNSAMPLE_PASS_HPP_INCLUDED
#define DOWNSAMPLE_PASS_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"

void downsample_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth_tex);

#endif

#ifndef UTIL_PASSES_HPP_INCLUDED
#define UTIL_PASSES_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"

void gen_perlin_noise2D(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image, uint32_t mip, uint32_t layer);
void gen_mipmaps(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId image);

#endif
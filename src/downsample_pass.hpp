#ifndef DOWNSAMPLE_PASS_HPP_INCLUDED
#define DOWNSAMPLE_PASS_HPP_INCLUDED

#include "rendergraph/rendergraph.hpp"

void downsample_depth(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth_tex);

struct DownsamplePass {
  DownsamplePass();

  void run(
    rendergraph::RenderGraph &graph,
    rendergraph::ImageResourceId src_normals,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId out_normals);

private:
  gpu::GraphicsPipeline downsample_gbuffer;
  gpu::GraphicsPipeline downsample_depth;
  VkSampler sampler;

  void run_downsample_gbuff(
    rendergraph::RenderGraph &graph,
    rendergraph::ImageResourceId src_normals,
    rendergraph::ImageResourceId depth,
    rendergraph::ImageResourceId out_normal);

  void run_downsample_depth(
    rendergraph::RenderGraph &graph,
    rendergraph::ImageResourceId depth,
    uint32_t src_mip);
  
};


#endif

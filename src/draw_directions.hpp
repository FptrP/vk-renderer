#ifndef DRAW_DIRECTIONS_HPP_INCLUDED
#define DRAW_DIRECTIONS_HPP_INCLUDED

#include "scene/camera.hpp"
#include "rendergraph/rendergraph.hpp"
#include "imgui_pass.hpp"

struct DrawDirs {
  DrawDirs() {
    pipeline = gpu::create_compute_pipeline();
    pipeline.set_program("rotations");
  }

  void query_angle() {
    ImGui::Begin("Angle");
    ImGui::SliderAngle("angle", &angle);
    ImGui::End();
  }

  void draw(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId out_image) {

    struct PassData {
      rendergraph::ImageViewId id;
    };

    graph.add_task<PassData>("draw_directions", 
      [&](PassData &out, rendergraph::RenderGraphBuilder &builder){
        out.id = builder.use_storage_image(out_image, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
      },
      [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
        auto set = resources.allocate_set(pipeline, 0);
    
        gpu::write_set(set,
          gpu::StorageTextureBinding {0, resources.get_view(input.id)}
        );

        const auto &extent = resources.get_image(input.id).get_extent();

        cmd.bind_pipeline(pipeline);
        cmd.bind_descriptors_compute(0, {set}, {});
        cmd.push_constants_compute(0, sizeof(angle), &angle);
        cmd.dispatch(extent.width/8, extent.height/4, 1);
      });

  }

private:
  float angle = 0;
  gpu::ComputePipeline pipeline;
};


#endif
#include "ssao.hpp"

#include "gpu/gpu.hpp"

#include <cstdlib>
#include <iostream>

rendergraph::ImageResourceId create_ssao_texture(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) {
  gpu::ImageInfo info {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  return graph.create_image(VK_IMAGE_TYPE_2D, info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
}

constexpr uint32_t SAMPLES_COUNT = 16;

struct SSAOParams {
  glm::mat4 projection;
  float fovy;
  float aspect;
  float znear;
  float zfar;
  glm::vec3 samples[SAMPLES_COUNT];
};

SSAOPass::SSAOPass(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId target) {
  
  pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("ssao");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({false, {graph.get_descriptor(target).format}});
  
  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  for (uint32_t i = 0; i < SAMPLES_COUNT; i++) {
    while (true) {
      float x = 2.f * rand()/float(RAND_MAX) - 1.f;
      float y = 2.f * rand()/float(RAND_MAX) - 1.f;
      float z = 2.f * rand()/float(RAND_MAX) - 1.f;
      float l2 = x*x + y*y + z*z;

      if (l2 < 1.0) {
        auto len = std::sqrt(l2);
        sphere_samples.push_back(glm::vec3 {x/len, y/len, z/len});
        break;
      }
    }

    auto val = sphere_samples.back();
    std::cout << val.x << " " << val.y << " " << val.z << "\n";
  }

}

void SSAOPass::draw(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId depth, rendergraph::ImageResourceId target, const SSAOInParams &params) {
  struct PassData {
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId rt;
  };
  
  graph.add_task<PassData>("SSAO",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.depth = builder.sample_image(depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.rt = builder.use_color_attachment(target, 0, 0);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      
      auto block = cmd.allocate_ubo<SSAOParams>();

      block.ptr->projection = params.projection;
      block.ptr->fovy = params.fovy;
      block.ptr->aspect = params.aspect;
      block.ptr->znear = params.znear;
      block.ptr->zfar = params.zfar;

      for (uint32_t i = 0; i < SAMPLES_COUNT; i++) {
        block.ptr->samples[i] = sphere_samples[i];
      }

      auto set = resources.allocate_set(pipeline, 0);
    
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {1, cmd.get_ubo_pool(), block});
      
      const auto &image_info = resources.get_image(input.rt).get_info();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_view(input.rt)});
      cmd.bind_pipeline(pipeline);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {block.offset});
      cmd.draw(3, 1, 0, 0);
      cmd.end_renderpass();
    }); 
}
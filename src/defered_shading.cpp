#include "defered_shading.hpp"
#include "gpu_transfer.hpp"

struct ShaderConstants {
  glm::mat4 inverse_camera;
  glm::mat4 shadow_mvp;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

DeferedShadingPass::DeferedShadingPass(rendergraph::RenderGraph &graph, SDL_Window *window) {
  auto format = graph.get_descriptor(graph.get_backbuffer()).format;
  
  pipeline = gpu::create_graphics_pipeline();
  pipeline.set_program("defered_shading");
  pipeline.set_registers({});
  pipeline.set_vertex_input({});
  pipeline.set_rendersubpass({false, {format}});

  imgui_init(window, pipeline.get_renderpass());

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  ubo_consts = graph.create_buffer(
    VMA_MEMORY_USAGE_GPU_ONLY, 
    sizeof(ShaderConstants), 
    VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void DeferedShadingPass::update_params(const glm::mat4 &camera, const glm::mat4 &shadow, float fovy, float aspect, float znear, float zfar) {
  ShaderConstants consts {
    glm::inverse(camera),
    shadow,
    fovy,
    aspect,
    znear,
    zfar 
  };

  gpu_transfer::write_buffer(ubo_consts, 0, sizeof(ShaderConstants), &consts);
}

void DeferedShadingPass::draw(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer, const rendergraph::ImageResourceId &shadow, const rendergraph::ImageResourceId &ssao, const rendergraph::ImageResourceId &out_image) {
  struct PassData {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId depth;
    rendergraph::ImageViewId rt;
    rendergraph::ImageViewId shadow;
    rendergraph::ImageViewId ssao;
    rendergraph::BufferResourceId ubo;
  };
  
  pipeline.set_rendersubpass({false, {graph.get_descriptor(out_image).format}});

  graph.add_task<PassData>("DeferedShading",
    [&](PassData &input, rendergraph::RenderGraphBuilder &builder){
      input.albedo = builder.sample_image(gbuffer.albedo, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.normal = builder.sample_image(gbuffer.normal, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.material = builder.sample_image(gbuffer.material, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.depth = builder.sample_image(gbuffer.depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.rt = builder.use_color_attachment(out_image, 0, 0);
      input.shadow = builder.sample_image(shadow, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.ssao = builder.sample_image(ssao, VK_SHADER_STAGE_FRAGMENT_BIT);
      input.ubo = ubo_consts;
      builder.use_uniform_buffer(input.ubo, VK_SHADER_STAGE_VERTEX_BIT);
    },
    [=](PassData &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      
      auto set = resources.allocate_set(pipeline, 0);
      
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.albedo), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.material), sampler},
        gpu::TextureBinding {3, resources.get_view(input.depth), sampler},
        gpu::UBOBinding {4, resources.get_buffer(input.ubo)}, 
        gpu::TextureBinding {5, resources.get_view(input.shadow), sampler},
        gpu::TextureBinding {6, resources.get_view(input.ssao), sampler});
      
      const auto &image_info = resources.get_image(input.rt).get_info();
      auto w = image_info.width;
      auto h = image_info.height;

      cmd.set_framebuffer(w, h, {resources.get_view(input.rt)});
      cmd.bind_pipeline(pipeline);
      cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
      cmd.bind_scissors(0, 0, w, h);
      cmd.bind_descriptors_graphics(0, {set}, {0});
      cmd.draw(3, 1, 0, 0);
      imgui_draw(cmd.get_command_buffer());
      cmd.end_renderpass();
    }); 
}
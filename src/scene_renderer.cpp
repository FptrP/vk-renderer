#include "scene_renderer.hpp"
#include "gpu_transfer.hpp"

#include <cstdlib>

Gbuffer::Gbuffer(rendergraph::RenderGraph &graph, uint32_t width, uint32_t height) : w {width}, h {height} {
  auto tiling = VK_IMAGE_TILING_OPTIMAL;
  auto color_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
  auto depth_usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;

  gpu::ImageInfo albedo_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo normal_info {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo mat_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
  gpu::ImageInfo depth_info {VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT, width, height};

  albedo = graph.create_image(VK_IMAGE_TYPE_2D, albedo_info, tiling, color_usage);
  normal = graph.create_image(VK_IMAGE_TYPE_2D, normal_info, tiling, color_usage);
  material = graph.create_image(VK_IMAGE_TYPE_2D, mat_info, tiling, color_usage);
  depth = graph.create_image(VK_IMAGE_TYPE_2D, depth_info, tiling, depth_usage);
}

void SceneRenderer::init_pipeline(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer) {
  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;
  
  opaque_pipeline = gpu::create_graphics_pipeline();
  opaque_pipeline.set_program("gbuf_opaque");
  opaque_pipeline.set_registers(regs);
  opaque_pipeline.set_vertex_input(scene::get_vertex_input());    
  
  opaque_pipeline.set_rendersubpass({true, {
    VK_FORMAT_R8G8B8A8_SRGB, 
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_D16_UNORM
  }});

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  view_proj_buffer = graph.create_buffer(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(glm::mat4), VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  transform_buffer = graph.create_buffer(VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(glm::mat4) * 1000, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

static void node_process(const scene::Node &node, std::vector<SceneRenderer::DrawCall> &draw_calls, std::vector<glm::mat4> &transforms, const glm::mat4 &acc) {
  auto transform = acc * node.transform;
  uint32_t transform_id = transforms.size();
  
  if (node.meshes.size()) {
    transforms.push_back(transform);
  }

  for (auto mesh_index : node.meshes) {
    draw_calls.push_back(SceneRenderer::DrawCall {transform_id, mesh_index});
  }

  for (auto &child : node.children) {
    node_process(*child, draw_calls, transforms, transform);
  }
}

void SceneRenderer::build_scene() {
  
}

void SceneRenderer::update_scene(const glm::mat4 &mvp) {
  auto identity = glm::identity<glm::mat4>();
  std::vector<glm::mat4> transforms;

  draw_calls.clear();
  
  node_process(*target.root, draw_calls, transforms, identity);

  gpu_transfer::write_buffer(view_proj_buffer, 0, sizeof(mvp), &mvp);
  gpu_transfer::write_buffer(transform_buffer, 0, sizeof(glm::mat4) * transforms.size(), transforms.data());
}

void SceneRenderer::draw(rendergraph::RenderGraph &graph, const Gbuffer &gbuffer) {
  
  struct Data {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId depth;
  };
  
  graph.add_task<Data>("GbufferPass",
    [&](Data &input, rendergraph::RenderGraphBuilder &builder){
      input.albedo = builder.use_color_attachment(gbuffer.albedo, 0, 0);
      input.normal = builder.use_color_attachment(gbuffer.normal, 0, 0);
      input.material = builder.use_color_attachment(gbuffer.material, 0, 0);
      input.depth = builder.use_depth_attachment(gbuffer.depth, 0, 0);

      builder.use_uniform_buffer(view_proj_buffer, VK_SHADER_STAGE_VERTEX_BIT);
      builder.use_storage_buffer(transform_buffer, VK_SHADER_STAGE_VERTEX_BIT);
    },
    [=](Data &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      cmd.set_framebuffer(gbuffer.w, gbuffer.h, {
        resources.get_view(input.albedo),
        resources.get_view(input.normal),
        resources.get_view(input.material),
        resources.get_view(input.depth)
      });

      auto vbuf = target.vertex_buffer.get_api_buffer();
      auto ibuf = target.index_buffer.get_api_buffer();
      
      cmd.bind_pipeline(opaque_pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.clear_depth_attachment(1.f);
      cmd.bind_viewport(0.f, 0.f, gbuffer.w, gbuffer.h, 0.f, 1.f);
      cmd.bind_scissors(0, 0, gbuffer.w, gbuffer.h);
      cmd.bind_vertex_buffers(0, {vbuf}, {0ul});
      cmd.bind_index_buffer(ibuf, 0, VK_INDEX_TYPE_UINT32);
      
      for (const auto &draw_call : draw_calls) {
        const auto &mesh = target.meshes[draw_call.mesh];
        const auto &material = target.materials[mesh.material_index];

        if (material.albedo_tex_index == scene::INVALID_TEXTURE) {
          continue;
        }

        auto set = resources.allocate_set(opaque_pipeline.get_layout(0));
        
        gpu::write_set(set, 
          gpu::UBOBinding {0, resources.get_buffer(view_proj_buffer)},
          gpu::SSBOBinding {1, resources.get_buffer(transform_buffer)},
          gpu::TextureBinding {2, target.images[material.albedo_tex_index], sampler});

        cmd.bind_descriptors_graphics(0, {set}, {0});
        cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t), &draw_call.transform);
        cmd.draw_indexed(mesh.index_count, 1, mesh.index_offset, mesh.vertex_offset, 0);
      }


      cmd.end_renderpass();
      
    });
}
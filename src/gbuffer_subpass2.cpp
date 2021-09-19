#include "gbuffer_subpass2.hpp"

GbufferData::GbufferData(gpu::PipelinePool &pipelines, rendergraph::RenderGraph &rendergraph, gpu::Device &device, uint32_t w, uint32_t h)
  : ubo {device.create_dynbuffer<GbufferShaderData>(rendergraph.get_frames_count())}
{

  auto tiling = VK_IMAGE_TILING_OPTIMAL;
  gpu::ImageInfo albedo_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
  gpu::ImageInfo normal_info {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
  gpu::ImageInfo mat_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, w, h};
  gpu::ImageInfo depth_info {VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT, w, h};

  albedo = rendergraph.create_image(VK_IMAGE_TYPE_2D, albedo_info, tiling, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
  normal = rendergraph.create_image(VK_IMAGE_TYPE_2D, normal_info, tiling, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
  material = rendergraph.create_image(VK_IMAGE_TYPE_2D, mat_info, tiling, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
  depth = rendergraph.create_image(VK_IMAGE_TYPE_2D, depth_info, tiling, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;
  
  gpu::VertexInput vinput;

  vinput.bindings = {{0, sizeof(scene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
  vinput.attributes = {
    {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32_SFLOAT,
      .offset = offsetof(scene::Vertex, pos)
    },
    {
      .location = 1,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32_SFLOAT,
      .offset = offsetof(scene::Vertex, norm)
    },
    {
      .location = 2,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = offsetof(scene::Vertex, uv)
    }
  };

  pipeline.attach(pipelines);
  pipeline.set_program("gbuf");
  pipeline.set_registers(regs);
  pipeline.set_vertex_input(vinput);    
  pipeline.set_rendersubpass({true, {albedo_info.format, normal_info.format, mat_info.format, depth_info.format}});

  width = w;
  height = h;
}

void add_gbuffer_subpass(GbufferData &gbuf, rendergraph::RenderGraph &rendergraph, scene::Scene &scene, glm::mat4 mvp) {

  struct GbuffResources {
    rendergraph::ImageViewId albedo;
    rendergraph::ImageViewId normal;
    rendergraph::ImageViewId material;
    rendergraph::ImageViewId depth;
  };

  rendergraph.add_task<GbuffResources>("GbufferPass",
    [&](GbuffResources &data, rendergraph::RenderGraphBuilder &builder){
      
      data.albedo = builder.use_color_attachment(gbuf.albedo, 0, 0);
      data.normal = builder.use_color_attachment(gbuf.normal, 0, 0);
      data.material = builder.use_color_attachment(gbuf.material, 0, 0);
      data.depth = builder.use_depth_attachment(gbuf.depth, 0, 0);

    },
    [=, &gbuf, &scene](GbuffResources &data, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
      
      auto set = resources.allocate_set(gbuf.pipeline.get_layout(0));
      gpu::DescriptorWriter writer {set};
      writer.bind_dynbuffer(0, gbuf.ubo);
      writer.write(resources.get_gpu().api_device());

      *gbuf.ubo.get_mapped_ptr(resources.get_frame_index()) = {mvp};

      uint32_t ubo_offset = gbuf.ubo.get_offset(resources.get_frame_index());

      cmd.set_framebuffer(gbuf.width, gbuf.height, {
        resources.get_view(data.albedo),
        resources.get_view(data.normal),
        resources.get_view(data.material),
        resources.get_view(data.depth)
      });

      auto vbuf = scene.get_vertex_buffer().get_api_buffer();
      auto ibuf = scene.get_index_buffer().get_api_buffer();
      const auto &mesh = scene.get_meshes().at(0);
      
      cmd.bind_pipeline(gbuf.pipeline);
      cmd.clear_color_attachments(0.f, 0.f, 0.f, 0.f);
      cmd.clear_depth_attachment(1.f);
      cmd.bind_descriptors_graphics(0, {set}, {ubo_offset});
      cmd.bind_viewport(0.f, 0.f, gbuf.width, gbuf.height, 0.f, 1.f);
      cmd.bind_scissors(0, 0, gbuf.width, gbuf.height);
      cmd.bind_vertex_buffers(0, {vbuf}, {0ul});
      cmd.bind_index_buffer(ibuf, 0, VK_INDEX_TYPE_UINT32);
      cmd.draw_indexed(mesh.index_count, 1, mesh.index_offset, mesh.vertex_offset, 0);
      cmd.end_renderpass();
    });

}
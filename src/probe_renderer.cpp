#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "probe_renderer.hpp"
#include <iostream>

ProbeRenderer::ProbeRenderer(rendergraph::RenderGraph &graph, uint32_t cubemap_res) {
  rendergraph::ImageDescriptor desc {};
  desc.type = VK_IMAGE_TYPE_2D;
  desc.height = cubemap_res;
  desc.width = cubemap_res;
  desc.format = VK_FORMAT_R8G8B8A8_SRGB;
  desc.array_layers = 6;
  desc.mip_levels = 1;
  desc.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  desc.tiling = VK_IMAGE_TILING_OPTIMAL;
  desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
  
  cubemap_color = graph.create_image(desc, gpu::ImageCreateOptions::Cubemap);
  
  desc.format = VK_FORMAT_R16_SFLOAT;
  cubemap_distance = graph.create_image(desc, gpu::ImageCreateOptions::Cubemap);

  gpu::ImageInfo depth_info {
    VK_FORMAT_D24_UNORM_S8_UINT, 
    VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT,
    cubemap_res,
    cubemap_res,
    1,
    1,
    1
  };

  rt_depth = graph.create_image(VK_IMAGE_TYPE_2D, depth_info, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  gpu::Registers regs {};
  regs.depth_stencil.depthTestEnable = VK_TRUE;
  regs.depth_stencil.depthWriteEnable = VK_TRUE;
  
  cubemap_pass = gpu::create_graphics_pipeline();
  cubemap_pass.set_program("cubemap_probe");
  cubemap_pass.set_registers(regs);
  cubemap_pass.set_vertex_input(scene::get_vertex_input());    
  
  cubemap_pass.set_rendersubpass({true, {
    VK_FORMAT_R8G8B8A8_SRGB, 
    VK_FORMAT_R16_SFLOAT,
    VK_FORMAT_D24_UNORM_S8_UINT
  }});

  octprobe_pass = gpu::create_compute_pipeline();
  octprobe_pass.set_program("cube2oct");

  downsample_pass = gpu::create_graphics_pipeline();
  downsample_pass.set_program("probe_downsample");
  downsample_pass.set_registers({});
  downsample_pass.set_vertex_input({});
  downsample_pass.set_rendersubpass({false, {VK_FORMAT_R16_UNORM}});

  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}

static glm::mat4 calc_matrix(uint32_t side, glm::vec3 pos) {
  //glm::mat4 proj = glm::perspective(glm::radians(90.f), aspect, 0.01f, 100.f);

  glm::vec3 fwd, up;
  switch(side) {
    case 0: fwd = glm::vec3{1.f, 0.f, 0.f}; up = glm::vec3{0.f, -1.f, 0.f}; break;
    case 1: fwd = glm::vec3{-1.f, 0.f, 0.f}; up = glm::vec3{0.f, -1.f, 0.f}; break;
    case 2: fwd = glm::vec3{0.f, 1.f, 0.f}; up = glm::vec3{0.f, 0.f, 1.f}; break;
    case 3: fwd = glm::vec3{0.f, -1.f, 0.f}; up = glm::vec3{0.f, 0.f, -1.f}; break;
    case 4: fwd = glm::vec3{0.f, 0.f, 1.f}; up = glm::vec3{0.f, -1.f, 0.f}; break;
    case 5: fwd = glm::vec3{0.f, 0.f, -1.f}; up = glm::vec3{0.f, -1.f, 0.f}; break;
  }
  return glm::lookAt(pos, pos + fwd, up);
}

void ProbeRenderer::render_cubemap(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, const glm::vec3 pos) {  
  for (uint32_t side = 0; side < 6; side++) {
    glm::mat4 view_mat = calc_matrix(side, pos);
    render_side(graph, scene_renderer, side, view_mat);
  }
}

void ProbeRenderer::render_side(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, uint32_t side, glm::mat4 view) {
  struct Res {
    rendergraph::ImageViewId color;
    rendergraph::ImageViewId distance;
    rendergraph::ImageViewId depth;
  };

  struct ShaderUbo {
    glm::mat4 projection;
    glm::mat4 camera;
  };

  struct PushData {
    uint32_t transform_index;
    uint32_t albedo_index;
  };

  //ShaderUbo ubo_data {view, glm::perspective(glm::radians(90.f), 1.f, 0.05f, 80.f)};
  /* TODO: Use new bindless textures
  graph.add_task<Res>("CubemapSide",
    [&](Res &res, rendergraph::RenderGraphBuilder &builder) {
      res.color = builder.use_color_attachment(cubemap_color, 0, side);
      res.distance = builder.use_color_attachment(cubemap_distance, 0, side);
      res.depth = builder.use_depth_attachment(rt_depth, 0, 0);

      builder.use_storage_buffer(scene_renderer.get_scene_transforms(), VK_SHADER_STAGE_VERTEX_BIT);
    },
    [=, &scene_renderer](Res &res, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      auto extent = resources.get_image(res.color).get_info().extent2D();

      auto blk = cmd.allocate_ubo<ShaderUbo>();
      glm::mat4 p = glm::perspective(glm::radians(90.f), 1.f, 0.05f, 80.f);
      blk.ptr->projection = p;
      blk.ptr->camera = view;

      cmd.set_framebuffer(extent.width, extent.height, {
        resources.get_view(res.color),
        resources.get_view(res.distance),
        resources.get_view(res.depth)
      });

      auto &target = scene_renderer.get_target(); 
      auto vbuf = target.vertex_buffer.get_api_buffer();
      auto ibuf = target.index_buffer.get_api_buffer();
      
      cmd.bind_pipeline(cubemap_pass);
      cmd.clear_depth_attachment(1.f);
      cmd.clear_color_attachments(100.f, 0.f, 0.f, 0.f);
      cmd.bind_viewport(0.f, 0.f, extent.width, extent.height, 0.f, 1.f);
      cmd.bind_scissors(0, 0, extent.width, extent.height);
      cmd.bind_vertex_buffers(0, {vbuf}, {0ul});
      cmd.bind_index_buffer(ibuf, 0, VK_INDEX_TYPE_UINT32);
      
      auto set = resources.allocate_set(cubemap_pass.get_layout(0));
      
      gpu::write_set(set, 
        gpu::UBOBinding {0, cmd.get_ubo_pool(), blk},
        gpu::SSBOBinding {1, resources.get_buffer(scene_renderer.get_scene_transforms())},
        gpu::ArrayOfImagesBinding {2, scene_renderer.get_images()},
        gpu::SamplerBinding {3, sampler});

      cmd.bind_descriptors_graphics(0, {set}, {blk.offset});

      for (const auto &draw_call : scene_renderer.get_drawcalls()) {
        const auto &mesh = target.meshes[draw_call.mesh];
        const auto &material = target.materials[mesh.material_index];

        if (material.albedo_tex_index == scene::INVALID_TEXTURE) {
          continue;
        }

        PushData pc {};
        pc.transform_index = draw_call.transform;
        pc.albedo_index = material.albedo_tex_index;
        
        cmd.push_constants_graphics(VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushData), &pc);
        cmd.draw_indexed(mesh.index_count, 1, mesh.index_offset, mesh.vertex_offset, 0);
      }


      cmd.end_renderpass();
    });*/
}

void ProbeRenderer::render_octahedral(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId probe_color, rendergraph::ImageResourceId probe_depth, uint32_t array_layer) {
  struct Input {
    rendergraph::ImageViewId cube_color;
    rendergraph::ImageViewId cube_distance;
    rendergraph::ImageViewId oct_color;
    rendergraph::ImageViewId oct_depth;
  };

  graph.add_task<Input>("Cubemap2Octahedral",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.cube_color = builder.sample_cubemap(cubemap_color, VK_SHADER_STAGE_COMPUTE_BIT);
      input.cube_distance = builder.sample_cubemap(cubemap_distance, VK_SHADER_STAGE_COMPUTE_BIT);
      input.oct_color = builder.use_storage_image(probe_color, VK_SHADER_STAGE_COMPUTE_BIT, 0, array_layer);
      input.oct_depth = builder.use_storage_image(probe_depth, VK_SHADER_STAGE_COMPUTE_BIT, 0, array_layer);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      const auto &desc = resources.get_image(input.oct_color)->get_extent();

      auto set = resources.allocate_set(octprobe_pass, 0);
      
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.cube_color), sampler},
        gpu::TextureBinding {1, resources.get_view(input.cube_distance), sampler},
        gpu::StorageTextureBinding {2, resources.get_view(input.oct_color)},
        gpu::StorageTextureBinding {3, resources.get_view(input.oct_depth)}
      );

      cmd.bind_pipeline(octprobe_pass);
      cmd.bind_descriptors_compute(0, {set});
      cmd.dispatch(desc.width/8, desc.height/4, 1);

    });
}

void ProbeRenderer::probe_downsample(rendergraph::RenderGraph &graph, rendergraph::ImageResourceId probe_depth, uint32_t array_layer) {
  auto desc = graph.get_descriptor(probe_depth);

  struct Input {
    rendergraph::ImageViewId depth_tex;
    rendergraph::ImageViewId depth_rt;
  };

  for (uint32_t i = 1; i < desc.mip_levels; i++) {
    graph.add_task<Input>("DownsampleProbe",
      [&](Input &input, rendergraph::RenderGraphBuilder &builder){
        input.depth_rt = builder.use_color_attachment(probe_depth, i, array_layer);
        input.depth_tex = builder.sample_image(probe_depth, VK_SHADER_STAGE_FRAGMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, array_layer, 1);
      },
      [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd){
        auto set = resources.allocate_set(downsample_pass, 0); 
        gpu::write_set(set, 
          gpu::TextureBinding {0, resources.get_view(input.depth_tex), sampler});

        uint32_t w = desc.width/(1 << i), h = desc.height/(1 << i);

        cmd.set_framebuffer(w, h, {resources.get_view(input.depth_rt)});
        cmd.bind_pipeline(downsample_pass);
        cmd.bind_descriptors_graphics(0, {set});
        cmd.bind_viewport(0.f, 0.f, float(w), float(h), 0.f, 1.f);
        cmd.bind_scissors(0, 0, w, h);
        cmd.draw(3, 1, 0, 0);
        cmd.end_renderpass();
      });
  }
}

void ProbeRenderer::render_probe(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, const glm::vec3 pos, OctahedralProbe &probe) {
  render_cubemap(graph, scene_renderer, pos);
  render_octahedral(graph, probe.color, probe.depth);
  probe_downsample(graph, probe.depth);
  probe.pos = pos;
}

static void swap_min(float &min, float &max) {
  float real_min = std::min(min, max);
  float real_max = std::max(min, max);
  min = real_min;
  max = real_max;
}

void ProbeRenderer::render_probe_grid(rendergraph::RenderGraph &graph, SceneRenderer &scene_renderer, glm::vec3 min, glm::vec3 max, OctahedralProbeGrid &probe_grid) {
  swap_min(min.x, max.x);
  swap_min(min.y, max.y);
  swap_min(min.z, max.z);
  
  probe_grid.min = min;
  probe_grid.max = max;

  if (probe_grid.grid_size < 2) {
    throw std::runtime_error {"Ooops"};
  }
  
  glm::vec3 step = (max - min)/float(probe_grid.grid_size - 1);
  for (uint32_t y = 0; y < probe_grid.grid_size; y++) {
    for (uint32_t x = 0; x < probe_grid.grid_size; x++) {
      glm::vec3 pos = min + step * glm::vec3{float(x), 0, float(y)};
      uint32_t array_layer = y * probe_grid.grid_size + x;

      render_cubemap(graph, scene_renderer, pos);
      render_octahedral(graph, probe_grid.color_array, probe_grid.depth_array, array_layer);
      probe_downsample(graph, probe_grid.depth_array, array_layer);

    }
  }
}

OctahedralProbe::OctahedralProbe(rendergraph::RenderGraph &graph, uint32_t size) {
  gpu::ImageInfo desc {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, size, size};
  color = graph.create_image(VK_IMAGE_TYPE_2D, desc, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t mips = std::floor(std::log2(size)) + 1;
  
  desc.mip_levels = mips;
  desc.format = VK_FORMAT_R16_UNORM;
  
  depth = graph.create_image(VK_IMAGE_TYPE_2D, desc, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
}

OctahedralProbeGrid::OctahedralProbeGrid(rendergraph::RenderGraph &graph, uint32_t grid_sz, uint32_t size)
  : grid_size {grid_sz}
{
  const uint32_t array_size = grid_sz * grid_sz;
  gpu::ImageInfo desc {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, size, size};
  
  desc.array_layers = array_size; 

  color_array = graph.create_image(VK_IMAGE_TYPE_2D, desc, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  const uint32_t mips = std::floor(std::log2(size)) + 1;
  
  desc.mip_levels = mips;
  desc.format = VK_FORMAT_R16_UNORM;
  
  depth_array = graph.create_image(VK_IMAGE_TYPE_2D, desc, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
}

ProbeTracePass::ProbeTracePass() {
  trace_pass = gpu::create_compute_pipeline();
  trace_pass.set_program("trace_probe");
  sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);
}

using ImageId = rendergraph::ImageResourceId;
using ImageViewId = rendergraph::ImageViewId;

void ProbeTracePass::run(
  rendergraph::RenderGraph &graph,
  OctahedralProbeGrid &probe,
  rendergraph::ImageResourceId gbuffer_depth,
  rendergraph::ImageResourceId gbuffer_norm,
  rendergraph::ImageResourceId out_image,
  const ProbeTraceParams &params)
{
  struct Input {
    ImageViewId depth;
    ImageViewId normal;
    ImageViewId probe_color;
    ImageViewId probe_depth;
    ImageViewId out_tex;
  };

  struct Constants {
    glm::mat4 inverse_view;
    glm::vec4 probe_min;
    glm::vec4 probe_max;
    uint32_t grid_size;
    float fovy;
    float aspect;
    float znear;
    float zfar;
  };

  Constants consts {
    params.inv_view,
    glm::vec4 {probe.min, 1.f},
    glm::vec4 {probe.max, 1.f},
    probe.grid_size,
    params.fovy,
    params.aspect,
    params.znear,
    params.zfar
  };

  graph.add_task<Input>("TraceProbe",
    [&](Input &input, rendergraph::RenderGraphBuilder &builder) {
      input.depth = builder.sample_image(gbuffer_depth, VK_SHADER_STAGE_COMPUTE_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1);
      input.normal = builder.sample_image(gbuffer_norm, VK_SHADER_STAGE_COMPUTE_BIT);
      input.probe_color = builder.sample_image(probe.color_array, VK_SHADER_STAGE_COMPUTE_BIT);
      input.probe_depth = builder.sample_image(probe.depth_array, VK_SHADER_STAGE_COMPUTE_BIT);
      input.out_tex = builder.use_storage_image(out_image, VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
    },
    [=](Input &input, rendergraph::RenderResources &resources, gpu::CmdContext &cmd) {
      const auto desc = resources.get_image(input.out_tex)->get_extent();
      auto blk = cmd.allocate_ubo<Constants>();
      *blk.ptr = consts;

      auto set = resources.allocate_set(trace_pass, 0);
      
      gpu::write_set(set,
        gpu::TextureBinding {0, resources.get_view(input.depth), sampler},
        gpu::TextureBinding {1, resources.get_view(input.normal), sampler},
        gpu::TextureBinding {2, resources.get_view(input.probe_color), sampler},
        gpu::TextureBinding {3, resources.get_view(input.probe_depth), sampler},
        //gpu::StorageTextureBinding {2, resources.get_view(input.oct_color)},
        gpu::UBOBinding {4, cmd.get_ubo_pool(), blk},
        gpu::StorageTextureBinding {5, resources.get_view(input.out_tex)}
      );

      cmd.bind_pipeline(trace_pass);
      cmd.bind_descriptors_compute(0, {set}, {blk.offset});
      cmd.dispatch(desc.width/8, desc.height/4, 1);

    });

}
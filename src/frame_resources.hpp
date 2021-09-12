#ifndef FRAME_RESOURCES_HPP_INCLUDED
#define FRAME_RESOURCES_HPP_INCLUDED

#include "framegraph/framegraph.hpp"
#include "gpu_context.hpp"
#include "scene/camera.hpp"
#include "scene/scene.hpp"

struct FrameResources {
  FrameResources(framegraph::RenderGraph &render_graph)
    : gbuf_depth {render_graph.create_image_desc(1, 1, VK_IMAGE_ASPECT_DEPTH_BIT, "gbuf_depth")},
      gbuf_albedo {render_graph.create_image_desc(1, 1, VK_IMAGE_ASPECT_COLOR_BIT, "gbuf_albedo")},
      gbuf_normal {render_graph.create_image_desc(1, 1, VK_IMAGE_ASPECT_COLOR_BIT, "gbuf_normal")},
      gbuf_material {render_graph.create_image_desc(1, 1, VK_IMAGE_ASPECT_COLOR_BIT, "gbuf_material")},
      backbuffer {render_graph.create_image_desc(1, 1, VK_IMAGE_ASPECT_COLOR_BIT, "backbuffer", true)}
  {} 


  uint32_t gbuf_depth;
  uint32_t gbuf_albedo;
  uint32_t gbuf_normal;
  uint32_t gbuf_material;
  uint32_t backbuffer;
};

struct GBuffer {
  GBuffer(gpu::Device &device) 
    : albedo {device.new_image()}, normal {device.new_image()}, 
      material {device.new_image()}, depth {device.new_image()} {}

  void create_images(uint32_t width, uint32_t height) {
    auto tiling = VK_IMAGE_TILING_OPTIMAL;
    gpu::ImageInfo albedo_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
    gpu::ImageInfo normal_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
    gpu::ImageInfo mat_info {VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, width, height};
    gpu::ImageInfo depth_info {VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT, width, height};

    albedo.create(VK_IMAGE_TYPE_2D, albedo_info, tiling, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    normal.create(VK_IMAGE_TYPE_2D, normal_info, tiling, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    material.create(VK_IMAGE_TYPE_2D, mat_info, tiling, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    depth.create(VK_IMAGE_TYPE_2D, depth_info, tiling, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
  }

  VkExtent3D get_ext_layers() const {
    auto ext = albedo.get_extent();
    return {ext.width, ext.height, 1};
  }

  VkExtent2D get_extent() const {
    auto ext = albedo.get_extent();
    return {ext.width, ext.height};
  }

  gpu::Image albedo;
  gpu::Image normal;
  gpu::Image material;
  gpu::Image depth;
};

struct FrameGlobal {
  FrameGlobal(gpu::Device &device, uint32_t width, uint32_t height, uint32_t max_frames_in_f) 
    : gbuffer {device}, frames_in_flight {max_frames_in_f}
  {
    projection = glm::perspective(glm::radians(60.f), float(width)/float(height), 0.01f, 10.f);
    gbuffer.create_images(width, height);
  }
  
  void process_event(const SDL_Event& e) {
    camera.process_event(e);
  }
  
  void bind_images(framegraph::RenderGraph &graph, const FrameResources &res) {
    graph.set_api_image(res.gbuf_albedo, gbuffer.albedo.get_image());
    graph.set_api_image(res.gbuf_normal, gbuffer.normal.get_image());
    graph.set_api_image(res.gbuf_material, gbuffer.material.get_image());
    graph.set_api_image(res.gbuf_depth, gbuffer.depth.get_image());
  }

  void update(uint32_t frame, uint32_t backbuffer, float dt) {
    frame_index = frame;
    backbuffer_index = backbuffer;
    camera.move(dt);
    
    auto view = camera.get_view_mat();
    view_proj = projection * view;
    normal_mat = glm::inverse(view);
  }

  GBuffer gbuffer;
  
  uint32_t frame_index;
  uint32_t backbuffer_index;
  uint32_t frames_in_flight;

  scene::Camera camera;
  glm::mat4 projection;
  glm::mat4 view_proj;
  glm::mat4 normal_mat;
};


#endif
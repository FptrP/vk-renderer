#ifndef FRAME_RESOURCES_HPP_INCLUDED
#define FRAME_RESOURCES_HPP_INCLUDED

#include "framegraph/framegraph.hpp"
#include "gpu_context.hpp"
#include "scene/camera.hpp"

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

struct FrameGlobal {
  FrameGlobal(float width, float height) {
    projection = glm::perspective(glm::radians(60.f), width/height, 0.01f, 10.f);
  }
  
  void process_event(const SDL_Event& e) {
    camera.process_event(e);
  }
  
  void update(uint32_t frame, uint32_t backbuffer, float dt) {
    frame_index = frame;
    backbuffer_index = backbuffer;
    camera.move(dt);
  }

  //gpu::Image gbuf_depth;
  //gpu::Image gbuf_albedo;
  
  uint32_t frame_index;
  uint32_t backbuffer_index;
  
  scene::Camera camera;
  glm::mat4 projection;
};


#endif
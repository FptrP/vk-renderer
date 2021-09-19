#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <memory>

#include "gpu/driver.hpp"
#include "gpu/shader.hpp"
#include "scene/scene.hpp"
#include "base_app.hpp"
#include "gpu/imgui_context.hpp"
#include "gpu/pipelines.hpp"

#include "rendergraph/rendergraph.hpp"
#include "backbuffer_subpass2.hpp"
#include "gbuffer_subpass2.hpp"

struct RGApp : SDLVulkanAppBase {
  RGApp(uint32_t w, uint32_t h) 
    : SDLVulkanAppBase {w, h}, pipelines {gpu_device().api_device()}, render_graph {gpu_device(), gpu_swapchain()},
      scene {gpu_device()}
  {
    scene.load("assets/gltf/suzanne/Suzanne.gltf", "assets/gltf/suzanne/");
    scene.gen_buffers(gpu_device());
  }
  ~RGApp() {
    vkDeviceWaitIdle(gpu_device().api_device());
  }

  rendergraph::RenderGraph &get_graph() { return render_graph; }

  void submit() {
    render_graph.submit();
  }

  gpu::PipelinePool &get_pipelines() { return pipelines; }
  scene::Scene &get_scene() { return scene; }
private:
  gpu::PipelinePool pipelines;
  rendergraph::RenderGraph render_graph;
  scene::Scene scene;
  scene::Camera camera;

  glm::mat4 projection;
  glm::mat4 view_proj;
}; 

#include <iostream>

constexpr VkSamplerCreateInfo default_sampler {
  .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
  .pNext = nullptr,
  .flags = 0,
  .magFilter = VK_FILTER_LINEAR,
  .minFilter = VK_FILTER_LINEAR,
  .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  .mipLodBias = 0.f,
  .anisotropyEnable = VK_FALSE,
  .maxAnisotropy = 0.f,
  .compareEnable = VK_FALSE,
  .compareOp = VK_COMPARE_OP_ALWAYS,
  .minLod = 0.f,
  .maxLod = 10.f,
  .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
  .unnormalizedCoordinates = VK_FALSE
};

int main() {
  RGApp app {800, 600};
  auto &pool = app.get_pipelines();
  pool.create_program("triangle", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/triangle/vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/triangle/frag.spv", "main"}});
  
  pool.create_program("texdraw", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/texdraw/vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/texdraw/frag.spv", "main"}});
  
  pool.create_program("gbuf", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/gbuf/default_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/gbuf/default_frag.spv", "main"}});

  GbufferData gbuffer {pool, app.get_graph(), app.get_context().device, 800, 600};
  auto sampler = app.get_context().device.create_sampler(default_sampler);

  scene::Camera camera;
  glm::mat4 projection = glm::perspective(glm::radians(60.f), 800.f/600.f, 0.01f, 10.f);
  glm::mat4 mvp;
  
  bool quit = false;
  while (!quit) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        quit = true;
      }

      camera.process_event(event);
    }

    camera.move(1.f/30.f);
    mvp = projection * camera.get_view_mat();
    add_gbuffer_subpass(gbuffer, app.get_graph(), app.get_scene(), mvp);
    add_backbuffer_subpass(gbuffer.normal, sampler, app.get_graph(), pool);
    app.submit(); 
  }
  return 0;
}
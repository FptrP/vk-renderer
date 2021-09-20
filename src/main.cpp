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
#include "gpu/samplers.hpp"

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

  gpu::SamplerPool samplers {app.get_context().device.api_device()};

  GbufferData gbuffer {pool, app.get_graph(), app.get_context().device, 800, 600};
  auto sampler = samplers.get_sampler(gpu::DEFAULT_SAMPLER);

  scene::Camera camera;
  glm::mat4 projection = glm::perspective(glm::radians(60.f), 800.f/600.f, 0.01f, 10.f);
  glm::mat4 mvp = projection * camera.get_view_mat();
  
  bool quit = false;

  auto ticks = SDL_GetTicks();

  while (!quit) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        quit = true;
      }

      camera.process_event(event);
    }

    auto ticks_now = SDL_GetTicks();
    float dt = (ticks_now - ticks)/1000.f;
    ticks = ticks_now;

    camera.move(dt);


    mvp = projection * camera.get_view_mat();
    add_gbuffer_subpass(gbuffer, app.get_graph(), app.get_scene(), mvp);
    add_backbuffer_subpass(gbuffer.normal, sampler, app.get_graph(), pool);
    app.submit(); 
  }

  vkDeviceWaitIdle(app.get_context().device.api_device());


  return 0;
}
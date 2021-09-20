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
#include "util_passes.hpp"

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

  pool.create_program("perlin", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/perlin/vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/perlin/frag.spv", "main"}
  });

  gpu::SamplerPool samplers {app.get_context().device.api_device()};

  GbufferData gbuffer {pool, app.get_graph(), app.get_context().device, 800, 600};
  auto sampler = samplers.get_sampler(gpu::DEFAULT_SAMPLER);

  auto noise_image = app.get_graph().create_image(
    VK_IMAGE_TYPE_2D, {
      VK_FORMAT_R8G8B8A8_SRGB,
      VK_IMAGE_ASPECT_COLOR_BIT,
      256,
      256,
      1,
      8,
      1
    },
    VK_IMAGE_TILING_OPTIMAL, 
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  gen_perlin_noise2D(app.get_graph(), noise_image, app.get_pipelines(), 0, 0);
  gen_mipmaps(app.get_graph(), noise_image, app.get_pipelines());

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
    add_gbuffer_subpass(gbuffer, app.get_graph(), app.get_scene(), mvp, noise_image, sampler);
    add_backbuffer_subpass(gbuffer.albedo, sampler, app.get_graph(), pool);
    app.submit(); 
  }

  vkDeviceWaitIdle(app.get_context().device.api_device());


  return 0;
}
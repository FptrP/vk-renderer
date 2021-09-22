#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <memory>

#include "gpu/gpu.hpp"
#include "scene/scene.hpp"

#include "rendergraph/rendergraph.hpp"
#include "backbuffer_subpass2.hpp"
#include "gbuffer_subpass2.hpp"
#include "util_passes.hpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* pUserData)  
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

struct AppInit {
  AppInit(uint32_t width, uint32_t height) {
    SDL_Init(SDL_INIT_EVERYTHING);
    window = SDL_CreateWindow("T", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_VULKAN);

    uint32_t count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr);
    std::vector<const char*> ext; 
    ext.resize(count);
    SDL_Vulkan_GetInstanceExtensions(window, &count, ext.data());

    gpu::InstanceConfig instance_info {};
    instance_info.layers = {"VK_LAYER_KHRONOS_validation"};
    instance_info.extensions.insert(ext.begin(), ext.end());
    instance_info.extensions.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    gpu::init_instance(instance_info, debug_cb);

    gpu::DeviceConfig device_info {};
    device_info.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    gpu::init_device(device_info, {width, height}, [&](VkInstance instance){
      VkSurfaceKHR surface;
      SDL_Vulkan_CreateSurface(window, instance, &surface);
      return surface;
    });
  }

  ~AppInit() {
    gpu::close();
    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  SDL_Window *window;
};

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

int main() {
  AppInit app_init {WIDTH, HEIGHT};

  gpu::create_program("triangle", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/triangle/vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/triangle/frag.spv", "main"}});
  
  gpu::create_program("texdraw", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/texdraw/vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/texdraw/frag.spv", "main"}});
  
  gpu::create_program("gbuf", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/gbuf/default_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/gbuf/default_frag.spv", "main"}});

  gpu::create_program("perlin", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/perlin/vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/perlin/frag.spv", "main"}
  });

  rendergraph::RenderGraph render_graph {gpu::app_device(), gpu::app_swapchain()};
  auto transfer_pool = gpu::app_device().new_transfer_pool();

  auto scene = scene::load_gltf_scene(gpu::app_device(), transfer_pool, "assets/gltf/suzanne/Suzanne.gltf", "assets/gltf/suzanne/");

  GbufferData gbuffer {render_graph, WIDTH, HEIGHT};
  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  auto noise_image = render_graph.create_image(
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

  gen_perlin_noise2D(render_graph, noise_image, 0, 0);
  gen_mipmaps(render_graph, noise_image);

  scene::Camera camera;
  glm::mat4 projection = glm::perspective(glm::radians(60.f), float(WIDTH)/HEIGHT, 0.01f, 10.f);
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
    add_gbuffer_subpass(gbuffer, render_graph, scene, mvp, noise_image, sampler);
    add_backbuffer_subpass(render_graph, gbuffer.albedo, sampler);
    render_graph.submit(); 
  }

  vkDeviceWaitIdle(gpu::app_device().api_device());

  return 0;
}
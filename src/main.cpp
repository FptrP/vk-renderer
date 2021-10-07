#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <memory>

#include "gpu/gpu.hpp"
#include "scene/scene.hpp"
#include "rendergraph/rendergraph.hpp"

#include "backbuffer_subpass2.hpp"
#include "util_passes.hpp"
#include "scene_renderer.hpp"
#include "defered_shading.hpp"
#include "gpu_transfer.hpp"
#include "ssao.hpp"
#include "downsample_pass.hpp"
#include "ssr.hpp"

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

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

int main() {
  AppInit app_init {WIDTH, HEIGHT};

  gpu::create_program("triangle", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/triangle/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/triangle/shader_frag.spv", "main"}});
  
  gpu::create_program("texdraw", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/texdraw/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/texdraw/shader_frag.spv", "main"}});
  
  gpu::create_program("perlin", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/perlin/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/perlin/shader_frag.spv", "main"}
  });

  gpu::create_program("gbuf_opaque", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/gbuf/opaque_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/gbuf/opaque_frag.spv", "main"}
  });

  gpu::create_program("defered_shading", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/defered_shading/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/defered_shading/shader_frag.spv", "main"}
  });

  gpu::create_program("default_shadow", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/shadows/default_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/shadows/default_frag.spv", "main"}
  });
  
  gpu::create_program("ssao", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/ssao/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/ssao/shader_frag.spv", "main"}
  });

  gpu::create_program("downsample_depth", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/depth_downsample/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/depth_downsample/shader_frag.spv", "main"}
  });

  gpu::create_program("ssr", {
    {VK_SHADER_STAGE_VERTEX_BIT, "src/shaders/ssr/shader_vert.spv", "main"},
    {VK_SHADER_STAGE_FRAGMENT_BIT, "src/shaders/ssr/shader_frag.spv", "main"}
  });
  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  rendergraph::RenderGraph render_graph {gpu::app_device(), gpu::app_swapchain()};
  gpu_transfer::init(render_graph);

  auto transfer_pool = gpu::app_device().new_transfer_pool();
  auto scene = scene::load_gltf_scene(gpu::app_device(), transfer_pool, "assets/gltf/Sponza/glTF/Sponza.gltf", "assets/gltf/Sponza/glTF/");

  Gbuffer gbuffer {render_graph, WIDTH, HEIGHT};
  auto ssao_texture = create_ssao_texture(render_graph, WIDTH, HEIGHT); 
  auto ssr_texture = create_ssr_tex(render_graph, WIDTH, HEIGHT);

  SceneRenderer scene_renderer {scene};
  scene_renderer.init_pipeline(render_graph, gbuffer);
  DeferedShadingPass shading_pass {render_graph, app_init.window};

  imgui_create_fonts(transfer_pool);

  SSAOPass ssao_pass {render_graph, ssao_texture};

  auto shadows_tex = render_graph.create_image(
    VK_IMAGE_TYPE_2D, 
    gpu::ImageInfo{
      VK_FORMAT_D24_UNORM_S8_UINT, 
      VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT, 
      1024, 1024, 1, 1, 4
    }, 
    VK_IMAGE_TILING_OPTIMAL, 
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  scene::Camera camera;
  glm::mat4 projection = glm::perspective(glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f);
  glm::mat4 shadow_mvp = glm::perspective(glm::radians(90.f), 1.f, 0.05f, 80.f) * glm::lookAt(glm::vec3{0, 2, -1}, glm::vec3{0, 2, 1}, glm::vec3{0, -1, 0});

  bool quit = false;
  auto ticks = SDL_GetTicks();

  while (!quit) {
    imgui_new_frame();
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      imgui_handle_event(event);
      if (event.type == SDL_QUIT) {
        quit = true;
      }

      camera.process_event(event);
    }

    auto ticks_now = SDL_GetTicks();
    float dt = (ticks_now - ticks)/1000.f;
    ticks = ticks_now;

    camera.move(dt);
    scene_renderer.update_scene(camera.get_view_mat(), projection);
    shading_pass.update_params(camera.get_view_mat(), shadow_mvp, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f);
    
    gpu_transfer::process_requests(render_graph);
    scene_renderer.draw(render_graph, gbuffer);
    scene_renderer.render_shadow(render_graph, shadow_mvp, shadows_tex, 0);
    downsample_depth(render_graph, gbuffer.depth);
    ssao_pass.draw(render_graph, gbuffer.depth, ssao_texture, SSAOInParams {projection, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f});
    
    add_ssr_pass(render_graph, gbuffer.depth, gbuffer.normal, gbuffer.albedo, ssr_texture, SSRParams {
      glm::transpose(glm::inverse(camera.get_view_mat())),
      glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f
    });

    //shading_pass.draw(render_graph, gbuffer, shadows_tex, ssao_texture, render_graph.get_backbuffer());
    add_backbuffer_subpass(render_graph, ssr_texture, sampler);
    add_present_subpass(render_graph);
    render_graph.submit();  
  }
  
  vkDeviceWaitIdle(gpu::app_device().api_device());
  gpu_transfer::close();
  imgui_close();
  return 0;
}
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <lib/json.hpp>

using json = nlohmann::json; 
namespace fs = std::filesystem;

#include "gpu/gpu.hpp"
#include "scene/scene.hpp"
#include "scene/scene_as.hpp"
#include "rendergraph/rendergraph.hpp"

#include "backbuffer_subpass2.hpp"
#include "util_passes.hpp"
#include "scene_renderer.hpp"
#include "probe_renderer.hpp"
#include "defered_shading.hpp"
#include "gpu_transfer.hpp"
#include "ssao.hpp"
#include "downsample_pass.hpp"
#include "ssr.hpp"
#include "gtao.hpp"
#include "trace_samples.hpp"
#include "draw_directions.hpp"
#include "screen_trace.hpp"
#include "image_readback.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <lib/stb_image_write.h>

#define ENABLE_VALIDATION 1
#define USE_RAY_QUERY 0

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
  AppInit(uint32_t width, uint32_t height, bool enable_validation) {
    SDL_Init(SDL_INIT_EVERYTHING);
    window = SDL_CreateWindow("T", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_VULKAN);

    uint32_t count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr);
    std::vector<const char*> ext; 
    ext.resize(count);
    SDL_Vulkan_GetInstanceExtensions(window, &count, ext.data());

    gpu::InstanceConfig instance_info {};
    instance_info.api_version = VK_API_VERSION_1_2;
    if (enable_validation) {
      instance_info.layers = {"VK_LAYER_KHRONOS_validation"};
    }
    instance_info.extensions.insert(ext.begin(), ext.end());
    instance_info.extensions.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    gpu::DeviceConfig device_info {};
#if USE_RAY_QUERY
    device_info.use_ray_query = true;
#endif
    device_info.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    gpu::init_all(instance_info, debug_cb, device_info, {width, height}, [&](VkInstance instance){
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

void get_depth_cb(ReadBackData &&image) {
  const uint32_t *word_ptr = reinterpret_cast<const uint32_t*>(image.bytes.get());
  
  const char *save_path = "captures/gbuffer_depth.csv";
  std::ofstream file {save_path, std::ios::trunc};
  file << "y, ";

  for (uint32_t x = 0; x < image.width; x++) {
    file << x;
    if (x != image.width - 1) {
      file << ",";
    }
  }
  
  file << "\n";

  for (uint32_t y = 0; y < image.height; y++) {
    file << y << ",";
    file << std::hex;

    for (uint32_t x = 0; x < image.width; x++) {
      auto depth = 0xffffff & word_ptr[y * image.width + x];
      file << "0x" << depth;
      if (x != image.width - 1) {
        file << ",";
      }
    }
    
    file << std::dec;
    file << "\n";
  }
  file.close();
}

void get_depth_png_cb(ReadBackData &&image) {
  uint32_t *word_ptr = reinterpret_cast<uint32_t*>(image.bytes.get());
  for (uint32_t y = 0; y < image.height; y++) {
    for (uint32_t x = 0; x < image.width; x++) {
      word_ptr[y * image.width + x] &= 0xffffff;
    }
  }
  int res = stbi_write_png("captures/gbuffer_depth.png", int(image.width), int(image.height), 4, image.bytes.get(), 0);
  std::cout << "STBI: " << res << "\n";
}

void get_rgba_cb(ReadBackData &&image) {
  int res = stbi_write_png("captures/gbuffer_color.png", int(image.width), int(image.height), 4, image.bytes.get(), 0);
  std::cout << "STBI: " << res << "\n";
}

static void load_shaders(const fs::path &config_path) {
  const fs::path shader_dir = config_path.parent_path();

  std::ifstream conf_file {config_path};
  auto config = json::parse(conf_file);
  
  const std::unordered_map<std::string, VkShaderStageFlagBits> stages_map {
    {"vertex", VK_SHADER_STAGE_VERTEX_BIT},
    {"fragment", VK_SHADER_STAGE_FRAGMENT_BIT},
    {"compute", VK_SHADER_STAGE_COMPUTE_BIT}
  };

  for (const auto &elem : config.items()) {
    const auto prog_name = elem.key();
    const auto &prog = elem.value();

    std::vector<gpu::ShaderBinding> bindings; 
    bindings.reserve(prog.size());
    
    for (const auto &shader : prog.items()) {
      const auto &key = shader.key();
      const auto &val = shader.value();
      auto stage = stages_map.find(key); 
      if (stage == stages_map.end()) {
        throw std::runtime_error {"Incorrect stage"};
      }

      gpu::ShaderBinding binding;
      binding.stage = stage->second;
      binding.main = "main";

      auto file_path = shader_dir / val.get<std::string>();
      if (!file_path.has_extension()) {
        file_path += ".spv";
      }

      binding.path = file_path.string();
      //std::cout << "Loading " << binding.path << "\n";
      bindings.push_back(std::move(binding));
    }
    std::cout << "Loading " << prog_name << " program\n";
    gpu::create_program(prog_name, std::move(bindings));
  }
}

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

int main(int argc, char **argv) {
  bool enable_validation = true;
  
  std::vector<std::string> params;
  params.reserve(argc - 1);
  for (int i = 1; i < argc; i++) {
    params.push_back(argv[i]);
  }

  if (params.size() > 0 && params[0] == "--disable-validation") {
    std::cout << "validation disabled\n";
    enable_validation = false;
  }
  
  AppInit app_init {WIDTH, HEIGHT, enable_validation};
  load_shaders("src/shaders/config.json");

  auto sampler = gpu::create_sampler(gpu::DEFAULT_SAMPLER);

  rendergraph::RenderGraph render_graph {gpu::app_device(), gpu::app_swapchain()};
  gpu_transfer::init(render_graph);
  ReadBackSystem readback_system;

  gpu::TransferCmdPool transfer_pool {};
  auto scene = scene::load_gltf_scene(transfer_pool, "assets/gltf/Sponza/glTF/Sponza.gltf", "assets/gltf/Sponza/glTF/", USE_RAY_QUERY);

#if USE_RAY_QUERY
  scene::SceneAccelerationStructure acceleration_struct;
  acceleration_struct.build(transfer_pool, scene);
#endif

  SamplesMarker::init(render_graph, WIDTH, HEIGHT);
  
  Gbuffer gbuffer {render_graph, WIDTH, HEIGHT};
  GTAO gtao {render_graph, WIDTH, HEIGHT, USE_RAY_QUERY, 1};
  ScreenSpaceTrace screen_trace {render_graph, WIDTH, HEIGHT};
  ProbeRenderer probe_renderer {render_graph};
  ProbeTracePass probe_trace_pass {};
  auto ssr_texture = create_ssr_tex(render_graph, WIDTH, HEIGHT);

  SceneRenderer scene_renderer {scene};
  scene_renderer.init_pipeline(render_graph, gbuffer);
  DeferedShadingPass shading_pass {render_graph, app_init.window};

  imgui_create_fonts(transfer_pool);

  auto shadows_tex = render_graph.create_image(
    VK_IMAGE_TYPE_2D, 
    gpu::ImageInfo{
      VK_FORMAT_D24_UNORM_S8_UINT, 
      VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT, 
      1024, 1024, 1, 1, 4
    }, 
    VK_IMAGE_TILING_OPTIMAL, 
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  OctahedralProbe oct_probe {render_graph};

  scene::Camera camera({0.f, 1.f, -1.f});
  glm::mat4 projection = glm::perspective(glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f);
  glm::mat4 shadow_mvp = glm::perspective(glm::radians(90.f), 1.f, 0.05f, 80.f) * glm::lookAt(glm::vec3{-1.85867, 5.81832, -0.247114}, glm::vec3{0, 2, 1}, glm::vec3{0, -1, 0});

  bool quit = false;
  bool probe_rendered = false;
  auto ticks = SDL_GetTicks();
  
  clear_depth(render_graph, gbuffer.prev_depth);

  glm::mat4 prev_mvp = projection * camera.get_view_mat();
  ReadBackID image_read_back = INVALID_READBACK;

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

    SamplesMarker::clear(render_graph);

    scene_renderer.draw(render_graph, gbuffer);
    scene_renderer.render_shadow(render_graph, shadow_mvp, shadows_tex, 0);
    if (!probe_rendered) {
      probe_renderer.render_probe(render_graph, scene_renderer, glm::vec3 {-0.249147, 1.15185, -0.472075}, oct_probe);
    }
    
    downsample_depth(render_graph, gbuffer.depth);

    ImGui::Begin("Read texture");
    bool depth = ImGui::Button("Depth") && (image_read_back == INVALID_READBACK);
    bool color = ImGui::Button("Color") && (image_read_back == INVALID_READBACK);
    if (depth) {
      image_read_back = readback_system.read_image(render_graph, gbuffer.depth, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0);
    } else if (color) {
      image_read_back = readback_system.read_image(render_graph, gbuffer.albedo);
    }

    ImGui::End();

    auto normal_mat = glm::transpose(glm::inverse(camera.get_view_mat()));
    auto camera_to_world = glm::inverse(camera.get_view_mat());
    GTAOParams gtao_params {normal_mat, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    GTAORTParams gtao_rt_params {camera_to_world, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    GTAOReprojection gtao_reprojection {prev_mvp * glm::inverse(camera.get_view_mat()), glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    ScreenTraceParams trace_params {normal_mat, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f};
    ProbeTraceParams probe_trace_params {
      camera_to_world, glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f
    };
    //gtao.add_main_pass(render_graph, gtao_params, gbuffer.depth, gbuffer.normal);
    //gtao.add_main_rt_pass(render_graph, gtao_rt_params, acceleration_struct.tlas, gbuffer.depth, gbuffer.normal);
    screen_trace.add_main_pass(render_graph, trace_params, gbuffer.depth, gbuffer.normal, gbuffer.albedo, gbuffer.material);
    screen_trace.add_filter_pass(render_graph, trace_params, gbuffer.depth);
    screen_trace.add_accumulate_pass(render_graph, trace_params, gbuffer.depth, gbuffer.prev_depth);
    //gtao.add_filter_pass(render_graph, gtao_params, gbuffer.depth);
    //gtao.add_reprojection_pass(render_graph, gtao_reprojection, gbuffer.depth, gbuffer.prev_depth);
    //gtao.add_accumulate_pass(render_graph, gtao_reprojection, gbuffer.depth, gbuffer.prev_depth);

    /*add_ssr_pass(render_graph, gbuffer.depth, gbuffer.normal, gbuffer.albedo, gbuffer.material, ssr_texture, SSRParams {
      normal_mat,
      glm::radians(60.f), float(WIDTH)/HEIGHT, 0.05f, 80.f
    });*/

    probe_trace_pass.run(render_graph, oct_probe, gbuffer.depth, gbuffer.normal, ssr_texture, probe_trace_params);

    //shading_pass.draw(render_graph, gbuffer, shadows_tex, screen_trace.accumulated, render_graph.get_backbuffer());
    add_backbuffer_subpass(render_graph, ssr_texture, sampler, DrawTex::ShowAll);
    //add_backbuffer_subpass(render_graph, gtao.accumulated_ao, sampler, DrawTex::ShowR);
    
    add_present_subpass(render_graph);
    render_graph.submit();
    readback_system.after_submit(render_graph);

    if (image_read_back != INVALID_READBACK && readback_system.is_data_available(image_read_back)) {
      auto data = readback_system.get_data(image_read_back);
      if (data.texel_fmt == VK_FORMAT_D24_UNORM_S8_UINT) {
        get_depth_cb(std::move(data));
      } else {
        get_rgba_cb(std::move(data));
      }
      
      image_read_back = INVALID_READBACK;
    }

    render_graph.remap(gbuffer.depth, gbuffer.prev_depth);
    render_graph.remap(gtao.output, gtao.prev_frame);

    prev_mvp = projection * camera.get_view_mat();
  }
  
  vkDeviceWaitIdle(gpu::app_device().api_device());
  gpu_transfer::close();
  imgui_close();
  return 0;
}
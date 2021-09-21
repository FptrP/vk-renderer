#include "gpu.hpp"

#include <optional>
#include <memory>

namespace gpu {
  
  static std::optional<Instance> g_instance;
  static std::optional<DebugMessenger> g_messenger;
  static std::optional<Surface> g_surface;
  static std::optional<Device> g_device;
  static std::optional<Swapchain> g_swapchain;
  static std::unique_ptr<PipelinePool> g_pipeline_pool;
  static std::optional<SamplerPool> g_sampler_pool;

  void init_instance(const InstanceConfig &cfg, PFN_vkDebugUtilsMessengerCallbackEXT callback) {
    g_instance.emplace(Instance {cfg});

    if (callback) {
      g_messenger.emplace(g_instance->create_debug_messenger(callback));
    }
  }
  
  void init_device(DeviceConfig cfg, VkExtent2D window_size, SurfaceCreateCB &&surface_cb) {
    auto api_surface = surface_cb(g_instance->api_instance());
    g_surface.emplace(Surface {g_instance->api_instance(), api_surface});
    cfg.surface = g_surface->api_surface();
    g_device.emplace(Device {g_instance->api_instance(), cfg});
    
    g_swapchain.emplace(Swapchain {
      g_device->api_device(), 
      g_device->api_physical_device(),
      g_surface->api_surface(),
      window_size,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT});

    g_pipeline_pool.reset(new PipelinePool {g_device->api_device()});
    g_sampler_pool.emplace(SamplerPool {g_device->api_device()});
  }
  
  void close() {
    vkDeviceWaitIdle(g_device->api_device());

    auto ptr = g_pipeline_pool.get();
    delete ptr;

    g_pipeline_pool.release();
    g_sampler_pool.reset();
    g_swapchain.reset();
    g_device.reset();
    g_surface.reset();
    g_messenger.reset();
    g_instance.reset();
  }

  Instance &app_instance() {
    return g_instance.value();
  }

  Device &app_device() {
    return g_device.value();
  }

  Swapchain &app_swapchain() {
    return g_swapchain.value();
  }
  
  PipelinePool &app_pipelines() {
    return *g_pipeline_pool;
  }

  GraphicsPipeline create_graphics_pipeline() {
    return {g_pipeline_pool.get()};
  }

  ComputePipeline create_compute_pipeline() {
    return {g_pipeline_pool.get()};
  }

  Image create_image() {
    return g_device->new_image();
  }
  
  Buffer create_buffer() {
    return g_device->new_buffer();
  }

  VkSampler create_sampler(const VkSamplerCreateInfo &info) {
    return g_sampler_pool->get_sampler(info);
  }

  void create_program(const std::string &name, std::initializer_list<ShaderBinding> shaders) {
    g_pipeline_pool->create_program(name, shaders);
  }

}
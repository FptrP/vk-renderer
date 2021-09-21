#ifndef GPU_HPP_INCLUDED
#define GPU_HPP_INCLUDED

#include "driver.hpp"

#include <functional>

namespace gpu {

  using SurfaceCreateCB = std::function<VkSurfaceKHR(VkInstance)>;

  void init_instance(const InstanceConfig &cfg, PFN_vkDebugUtilsMessengerCallbackEXT callback = nullptr);
  void init_device(DeviceConfig cfg, VkExtent2D window_size, SurfaceCreateCB &&surface_cb);
  void close();

  Instance &app_instance();
  Device &app_device();
  Swapchain &app_swapchain();
  PipelinePool &app_pipelines();

  GraphicsPipeline create_graphics_pipeline();
  ComputePipeline create_compute_pipeline();

  Image create_image();
  Buffer create_buffer();
  VkSampler create_sampler(const VkSamplerCreateInfo &info);

  template<typename T>
  DynBuffer<T> create_dynbuffer(uint32_t elems_count) {
    return app_device().create_dynbuffer<T>(elems_count);
  }

  template <typename... Bindings> 
  void write_set(VkDescriptorSet set, const Bindings&... bindings) {
    internal::write_set(app_device(), set, bindings...);
  }

  void create_program(const std::string &name, std::initializer_list<ShaderBinding> shaders);
}

#endif
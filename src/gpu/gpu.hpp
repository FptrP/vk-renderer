#ifndef GPU_HPP_INCLUDED
#define GPU_HPP_INCLUDED

#include "lib/volk.h"
#include <lib/vk_mem_alloc.h>
#include <memory>
#include <set>
#include <string>
#include <stdexcept>

namespace gpu {

#define VKCHECK(expr) vk_check_error((expr), __FILE__, __LINE__, #expr) 

void vk_check_error(VkResult result, const char *file, int line, const char *cmd);

struct DeviceConfig {
  std::set<std::string> instance_ext;
  std::set<std::string> layers;
  std::set<std::string> device_ext;

  bool create_swapchain = false;
  bool create_debug_log = false;
  PFN_vkDebugUtilsMessengerCallbackEXT debug_log = nullptr;
  
  virtual VkSurfaceKHR create_surface(VkInstance) const {
    return nullptr;
  }

  virtual VkExtent2D get_extent() const { return {0, 0}; }
};


struct Device {
  Device(const DeviceConfig &config);
  ~Device();

  VkDevice get_device() const { return device; }
  VkQueue get_queue() const { return queue; }
  uint32_t get_queue_family() const { return queue_family; }

private:
  void create_instance(const DeviceConfig &config);
  void init_instance_extensions(const DeviceConfig &config);
  void find_device(const DeviceConfig &config);
  void create_device(const DeviceConfig &config);
  void create_swapchain(const DeviceConfig &config);
  void destroy_instance_extensions();
  
  VkInstance instance {};
  VkDebugUtilsMessengerEXT debug_messenger {};
  VkSurfaceKHR surface {};
  
  VkPhysicalDevice phys_device {};
  VkDevice device {};
  
  VkQueue queue {};
  uint32_t queue_family {0u};

  VkSwapchainKHR swapchain {};
  VkExtent2D swapchain_extent {};
  VkFormat swapchain_format {};
};

struct Instance {
  Instance(const DeviceConfig &config) : gpu {config} {}
  ~Instance() {}

private:
  Device gpu;
};


}

#endif
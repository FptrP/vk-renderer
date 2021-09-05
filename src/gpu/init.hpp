#ifndef INIT_HPP_INCLUDED
#define INIT_HPP_INCLUDED

#include "lib/volk.h"
#include <lib/vk_mem_alloc.h>
#include <memory>
#include <set>
#include <string>
#include <stdexcept>
#include <vector>

//#include "image.hpp"
#include "vkerror.hpp"

namespace gpu {

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
  VmaAllocator get_allocator() const { return allocator; }

  uint32_t get_swapchain_image_count() const { return swapchain_images.size(); }
  VkImage *get_swapchain_images() { return swapchain_images.data(); }
  VkSwapchainKHR get_swapchain() const { return swapchain; }
  VkFormat get_swapchain_format() const { return swapchain_format; }
  VkExtent2D get_swapchain_extent() const { return swapchain_extent; }
  
  //std::vector<gpu::Image> get_backbuffers();

private:
  void create_instance(const DeviceConfig &config);
  void init_instance_extensions(const DeviceConfig &config);
  void find_device(const DeviceConfig &config);
  void create_device(const DeviceConfig &config);
  void create_swapchain(const DeviceConfig &config);
  void destroy_instance_extensions();
  void init_allocator();

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
  std::vector<VkImage> swapchain_images;
   
  VmaAllocator allocator {};
};


}

#endif
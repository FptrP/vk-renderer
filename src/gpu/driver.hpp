#ifndef DRIVER_HPP_INCLUDED
#define DRIVER_HPP_INCLUDED

#include <lib/volk.h>
#include <lib/vk_mem_alloc.h>
#include <set>
#include <string>

#include "vkerror.hpp"
#include "resources.hpp"
#include "shader.hpp"
#include "cmd_buffers.hpp"
#include "sync_primitive.hpp"
#include "dynbuffer.hpp"
#include "samplers.hpp"
#include "descriptors.hpp"

namespace gpu {

  struct Device;
  struct DebugMessenger;
  struct Swapchain;
  struct Swapchain;

  struct InstanceConfig {
    std::string app_name = "NOAPP";
    std::string engine_name = "NOENGINE";
    
    uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);
    uint32_t engine_version = VK_MAKE_VERSION(1, 0, 0);
    uint32_t api_version = VK_API_VERSION_1_0;

    std::set<std::string> extensions;
    std::set<std::string> layers;
  };

  struct DeviceConfig {
    VkSurfaceKHR surface {nullptr};
    std::set<std::string> extensions;
  };

  struct Instance {
    Instance(const InstanceConfig &cfg);
    Instance(Instance &&instance) : handle {instance.handle} { instance.handle = nullptr; }
    ~Instance();

    Device create_device(const DeviceConfig &cfg);
    DebugMessenger create_debug_messenger(PFN_vkDebugUtilsMessengerCallbackEXT callback);

    const Instance &operator=(Instance &&instance) { std::swap(handle, instance.handle); return *this; }
    
    const Instance &operator=(const Instance &instance) = delete;
    Instance(const Instance&) = delete;
  
    VkInstance api_instance() const { return handle; }
  private:
    VkInstance handle {nullptr};
  };
  
  struct DebugMessenger {
    DebugMessenger(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT callback);
    ~DebugMessenger();

    DebugMessenger(DebugMessenger &&msg) : base {msg.base}, handle {msg.handle} { msg.base = nullptr; msg.handle = nullptr; }
    
    const DebugMessenger &operator=(DebugMessenger &&msg) { 
      std::swap(handle, msg.handle); 
      std::swap(base, msg.base);
      return *this;
    }

    DebugMessenger(const DebugMessenger&) = delete;
    const DebugMessenger &operator=(const DebugMessenger &msg) = delete;
  private:
    VkInstance base {nullptr};
    VkDebugUtilsMessengerEXT handle {nullptr};
  };

  struct Device {
    Device(VkInstance instance, const DeviceConfig &cfg);
    Device(Device &&dev);
    ~Device();

    const Device &operator=(Device &&dev);

    Device(Device&) = delete;
    const Device &operator=(const Device &dev) = delete;

    Image new_image() const { return {logical_device, allocator}; }
    Buffer new_buffer() const { return {allocator}; }
    DescriptorPool new_descriptor_pool(uint32_t flips_count) { return {logical_device, flips_count}; }
    CmdBufferPool new_command_pool() const { return CmdBufferPool {logical_device, queue_family_index}; }
    Semaphore new_semaphore() const { return {logical_device}; }
    Fence new_fence(bool signaled = false) const { return {logical_device, signaled}; }
    TransferCmdPool new_transfer_pool() const { return TransferCmdPool {logical_device, queue_family_index, queue}; }
    
    Swapchain create_swapchain(VkSurfaceKHR surface, VkExtent2D window_extent, VkImageUsageFlags usage);
    std::vector<Image> get_swapchain_images(const Swapchain &swapchain);

    VkDevice api_device() const { return logical_device; }
    VkQueue api_queue() const { return queue; }
    VkPhysicalDevice api_physical_device() const { return physical_device; }
    uint32_t get_queue_family() const { return queue_family_index; }

    template <typename T>
    DynBuffer<T> create_dynbuffer(uint32_t elems) const { return DynBuffer<T> {allocator, properties.limits.minUniformBufferOffsetAlignment, elems}; }

    Sampler create_sampler(VkSamplerCreateInfo info) const { return Sampler {logical_device, info}; }
    VmaAllocator get_allocator() const { return allocator; }

    std::vector<CmdContext> allocate_cmd_contexts(CmdBufferPool &pool, uint32_t count);

  private:
    VkPhysicalDevice physical_device {nullptr};
    VkPhysicalDeviceProperties properties;
    VkDevice logical_device {nullptr};
    VmaAllocator allocator {};

    uint32_t queue_family_index;
    VkQueue queue {nullptr};
  };

  struct Swapchain {
    Swapchain(VkDevice device, VkPhysicalDevice physical_device, VkSurfaceKHR surface, VkExtent2D window, VkImageUsageFlags image_usage);
    ~Swapchain();

    Swapchain(Swapchain &&o) 
      : base {o.base}, handle {o.handle}, descriptor {o.descriptor}
    {
      o.handle = nullptr;
    }

    const Swapchain &operator=(Swapchain &&o) {
      std::swap(base, o.base);
      std::swap(handle, o.handle);
      std::swap(descriptor, o.descriptor);
      return *this;
    }

    uint32_t get_images_count() const {
      uint32_t count;
      vkGetSwapchainImagesKHR(base, handle, &count, nullptr);
      return count;
    }

    VkSwapchainKHR api_swapchain() const { return handle; }
    const ImageInfo &get_image_info() const { return descriptor; }
  private:
    VkDevice base {nullptr};
    VkSwapchainKHR handle {nullptr};
    ImageInfo descriptor {};

    Swapchain(const Swapchain&) = delete;
    const Swapchain& operator=(const Swapchain&) = delete;
  };

  struct Surface {
    Surface(VkInstance instance, VkSurfaceKHR surface) : base {instance}, handle {surface} {}
    ~Surface() { 
      if (base && handle) {
        vkDestroySurfaceKHR(base, handle, nullptr);
      }
    }

    Surface(Surface &&o) : base {o.base}, handle {o.handle} { o.handle = nullptr; }

    VkSurfaceKHR api_surface() const { return handle; }

    const Surface &operator=(Surface &&o) {
      std::swap(base, o.base);
      std::swap(handle, o.handle);
      return *this;
    }

  private:
    VkInstance base {nullptr};
    VkSurfaceKHR handle {nullptr};

    Surface(const Surface&) = delete;
    const Surface &operator=(const Surface&) = delete;
  };

  template <typename... Bindings> 
  void write_set(Device &device, VkDescriptorSet set, const Bindings&... bindings) {
    internal::write_set(device.api_device(), set, bindings...);
  }

}

#endif
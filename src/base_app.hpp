#ifndef BASE_APP_HPP_INCLUDED
#define BASE_APP_HPP_INCLUDED

#include "gpu/driver.hpp"

#include <memory>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <iostream>

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                       VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                       void* pUserData)  
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

struct SDLVulkanAppBase {
  SDLVulkanAppBase(uint32_t width, uint32_t height) {
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

    gpu::Instance instance {instance_info};
    
    auto messenger = instance.create_debug_messenger(debug_callback);

    VkSurfaceKHR api_surface;
    SDL_Vulkan_CreateSurface(window, instance.api_instance(), &api_surface);

    gpu::Surface surface {instance.api_instance(), api_surface};

    gpu::DeviceConfig device_info {};
    device_info.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    device_info.surface = api_surface;

    gpu::Device device {instance.create_device(device_info)};
    
    auto swapchain = device.create_swapchain(surface.api_surface(), {width, height}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    auto images = device.get_swapchain_images(swapchain);
    
    api.reset(new Api {
      std::move(instance),
      std::move(messenger),
      std::move(surface),
      std::move(device),
      std::move(swapchain),
      std::move(images)
    });
  }

  ~SDLVulkanAppBase() {
    vkDeviceWaitIdle(api->device.api_device());
    api.release();
    SDL_DestroyWindow(window);
    SDL_Quit();
  }

  gpu::Device &gpu_device() { return api->device; }
  const gpu::Device &gpu_device() const { return api->device; }
  auto &gpu_swapchain() const { return api->swapchain; }
  std::vector<gpu::Image> &backbuffers() { return api->backbuffer_images; };
  const gpu::ImageInfo swapchain_fmt() const { return api->swapchain.get_image_info(); }

  SDL_Window *sdl_window() { return window; }
  
private:

  struct Api {
    gpu::Instance instance;
    gpu::DebugMessenger debug_messenger;
    gpu::Surface surface;
    gpu::Device device;
    gpu::Swapchain swapchain;
    std::vector<gpu::Image> backbuffer_images;
  };

  SDL_Window *window {nullptr};
  std::unique_ptr<Api> api;
};

#endif
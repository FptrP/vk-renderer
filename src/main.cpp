#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include "gpu/gpu.hpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                       VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                       void* pUserData)  
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

struct SDLConfig : public gpu::DeviceConfig {
  SDLConfig(SDL_Window *window) : wnd {window} {
    uint32_t count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr);
    std::vector<const char*> ext; 
    ext.resize(count);
    SDL_Vulkan_GetInstanceExtensions(window, &count, ext.data());
  
    instance_ext.insert(ext.begin(), ext.end());
    instance_ext.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    layers.insert("VK_LAYER_KHRONOS_validation");
    device_ext.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    create_swapchain = true;
    create_debug_log = true;
    debug_log = debug_callback;
  }

  VkSurfaceKHR create_surface(VkInstance instance) const override {
    VkSurfaceKHR surface {};
    SDL_Vulkan_CreateSurface(wnd, instance, &surface);
    return surface;
  }

  VkExtent2D get_extent() const override {
    int x, y;
    SDL_Vulkan_GetDrawableSize(wnd, &x, &y);
    return {(uint32_t)x, (uint32_t)y};
  }

private:
  SDL_Window *wnd {};
};

int main() {
  SDL_Init(SDL_INIT_VIDEO|SDL_INIT_EVENTS);

  auto window = SDL_CreateWindow("", 0, 0, 800, 600, SDL_WINDOW_VULKAN);
  
  {
    SDLConfig config {window};
    gpu::Instance d3d {config};

    bool quit = false;
    while (!quit) {
    SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
          quit = true;
        }

        SDL_Delay(10); 
      }
    }
  }
  
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
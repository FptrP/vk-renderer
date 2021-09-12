#ifndef GPU_CONTEXT_HPP
#define GPU_CONTEXT_HPP

#include "gpu/driver.hpp"
#include <SDL2/SDL.h>

struct GpuContext {
  gpu::Instance &instance;
  gpu::Device &device;
  gpu::Swapchain &swapchain;
  SDL_Window *window;
};

#endif
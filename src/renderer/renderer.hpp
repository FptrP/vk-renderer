#ifndef RENDERER_HPP_INCLUDED
#define RENDERER_HPP_INCLUDED

#include "gpu/driver.hpp"
#include "gpu/imgui_context.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

namespace renderer {

  struct Renderer {
    Renderer(SDL_Window *window, uint32_t width, uint32_t height);
    ~Renderer();

    void draw_frame();

  private:
    gpu::Instance instance;
    gpu::DebugMessenger debug_messenger;
    gpu::Surface surface;
    gpu::Device device;
    gpu::Swapchain swapchain;
    
    std::vector<gpu::Image> backbuffer_images;

    gpu::CmdBufferPool cmdbuffer_pool;

    struct FrameContext {
      VkCommandBuffer cmd_buffer;
      gpu::Semaphore image_acquire_sem;
      gpu::Semaphore submit_done_sem;
      gpu::Fence submit_done_fence; 
    };

    std::vector<FrameContext> frames;
    uint32_t frame_id;

    void write_draw_commands(VkCommandBuffer cmd, uint32_t backbuffer_index, uint32_t frame_index);

  };

}

#endif
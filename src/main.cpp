#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <memory>

#include "gpu/driver.hpp"
#include "gpu/shader.hpp"
#include "base_app.hpp"


struct App : SDLVulkanAppBase {
  App(uint32_t width, uint32_t height) 
    : SDLVulkanAppBase {width, height},
      desc_alloc {gpu_device().new_descriptor_pool(3)},
      main_subpass {gpu_device().api_device(), {swapchain_fmt().format}},
      triangle_pipeline {init_pipeline(gpu_device().api_device(), main_subpass)},
      cmdbuffer_pool {gpu_device().new_command_pool()},
      frames_count {(uint32_t)backbuffers().size()}
  {
    const auto &dev = gpu_device();
    cmd_buffers = cmdbuffer_pool.allocate(frames_count);
    
    for (uint32_t i = 0; i < frames_count; i++) {
      submit_fences.push_back(dev.new_fence(true));
      image_acquire_semaphores.push_back(dev.new_semaphore());  
      submit_done_semaphores.push_back(dev.new_semaphore());
    }

    auto ext = swapchain_fmt().extent3D();
    ext.depth = 1;

    for (auto &img : backbuffers()) {
      gpu::ImageViewRange view {VK_IMAGE_VIEW_TYPE_2D, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      framebuffers.push_back(gpu::Framebuffer {gpu_device().api_device(), main_subpass, ext, {img.get_view(view)}});
    }
  
  }
  
  void run() {
    bool quit = false; 

    while (!quit) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
          quit = true;
        }
      } 
    
      render();
    }

    vkDeviceWaitIdle(gpu_device().api_device());
  }

  void render() {
    VkCommandBuffer cmd = cmd_buffers[frame_index]; 
    VkDevice device = gpu_device().api_device();
    VkQueue queue = gpu_device().api_queueu();
    VkSwapchainKHR swapchain = gpu_swapchain().api_swapchain();
    VkFence cmd_fence = submit_fences[frame_index];

    uint32_t image_index = 0;
    VKCHECK(vkAcquireNextImageKHR(device, swapchain, ~0ull, image_acquire_semaphores[frame_index], nullptr, &image_index)); 

    gpu::Image &image = backbuffers()[image_index];

    vkWaitForFences(device, 1, &cmd_fence, VK_TRUE, ~0ull);
    submit_fences[frame_index].reset();
    vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_cmd {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    
    vkBeginCommandBuffer(cmd, &begin_cmd);
    
    gpu::Barrier to_attachment {
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      {
        gpu::ImageBarrier {image, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
      }
    };

    to_attachment.flush(cmd);

    VkRenderPassBeginInfo renderpass_begin {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .pNext = nullptr,
      .renderPass = main_subpass.api_renderpass(),
      .framebuffer = framebuffers[image_index].api_framebuffer(),
      .renderArea = {{0, 0}, swapchain_fmt().extent2D()},
      .clearValueCount = 0,
      .pClearValues = nullptr
    };

    vkCmdBeginRenderPass(cmd, &renderpass_begin, VK_SUBPASS_CONTENTS_INLINE);
    
    VkClearAttachment clear {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .colorAttachment = 0,
      .clearValue {.color {0.f, 1.f, 0.f}}
    };

    VkClearRect clear_rect {
      .rect {{0, 0}, swapchain_fmt().extent2D()},
      .baseArrayLayer = 0,
      .layerCount = 1
    };

    VkViewport vp {0.f, 0.f, (float)swapchain_fmt().width, (float)swapchain_fmt().height, 0.f, 1.f};

    vkCmdClearAttachments(cmd, 1, &clear, 1, &clear_rect);
    triangle_pipeline.bind(cmd);
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &clear_rect.rect);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    
    gpu::Barrier to_present {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      {
        gpu::ImageBarrier {image, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR}
      }
    };

    to_present.flush(cmd);

    vkEndCommandBuffer(cmd);

    VkSemaphore wait_sem = image_acquire_semaphores[frame_index];
    VkSemaphore signal_sem = submit_done_semaphores[frame_index];
    VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo submit_info {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &wait_sem,
      .pWaitDstStageMask = &wait_mask,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &signal_sem
    };

    VKCHECK(vkQueueSubmit(queue, 1, &submit_info, cmd_fence));

    VkResult present_result;

    VkPresentInfoKHR present_info {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &signal_sem,
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &image_index,
      .pResults = &present_result
    };

    VKCHECK(vkQueuePresentKHR(queue, &present_info));
    VKCHECK(present_result);

    frame_index = (frame_index + 1) % frames_count;
  }

private:
  gpu::DescriptorPool desc_alloc;
  gpu::RenderSubpass main_subpass;
  gpu::Pipeline triangle_pipeline;
  gpu::CmdBufferPool cmdbuffer_pool;
  std::vector<gpu::Framebuffer> framebuffers;

  const uint32_t frames_count = 0;

  std::vector<VkCommandBuffer> cmd_buffers;
  std::vector<gpu::Fence> submit_fences;
  std::vector<gpu::Semaphore> image_acquire_semaphores;
  std::vector<gpu::Semaphore> submit_done_semaphores;

  uint32_t frame_index = 0;

  static gpu::Pipeline init_pipeline(VkDevice device, const gpu::RenderSubpass &subpass) {
    gpu::ShaderModule vertex {device, "src/shaders/triangle/vert.spv", VK_SHADER_STAGE_VERTEX_BIT};
    gpu::ShaderModule fragment {device, "src/shaders/triangle/frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT};

    gpu::GraphicsPipelineDescriptor registers {1};

    gpu::Pipeline pipeline {device};
    pipeline.init_gfx(subpass, vertex, fragment, registers);
    return pipeline;
  }

};

int main() {
  App app {800, 600};
  app.run();
  return 0;
}
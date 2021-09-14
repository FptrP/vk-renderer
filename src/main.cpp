#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vector>
#include <iostream>
#include <memory>

#include "gpu/driver.hpp"
#include "gpu/shader.hpp"
#include "scene/scene.hpp"
#include "base_app.hpp"
#include "gpu/imgui_context.hpp"
#include "framegraph/framegraph.hpp"
#include "backbuffer_subpass.hpp"
#include "gbuffer_subpass.hpp"
#include "frame_resources.hpp"

#include "rendergraph/rendergraph.hpp"

struct App : SDLVulkanAppBase {
  App(uint32_t width, uint32_t height) 
    : SDLVulkanAppBase {width, height},
      desc_alloc {gpu_device().new_descriptor_pool((uint32_t)backbuffers().size())},
      cmdbuffer_pool {gpu_device().new_command_pool()},
      frames_count {(uint32_t)backbuffers().size()},
      render_graph {},
      frame_resources {render_graph},
      frame_state {gpu_device(), width, height, frames_count},
      gbuffer_subpass {get_context(), frame_state},
      backbuffer_subpass {render_graph, get_context()}
  {
    const auto &dev = gpu_device();
    cmd_buffers = cmdbuffer_pool.allocate(frames_count);
    
    for (uint32_t i = 0; i < frames_count; i++) {
      submit_fences.push_back(dev.new_fence(true));
      image_acquire_semaphores.push_back(dev.new_semaphore());  
      submit_done_semaphores.push_back(dev.new_semaphore());
    }

    frame_state.bind_images(render_graph, frame_resources);
    
    backbuffer_subpass.create_fonts(cmd_buffers[0]);
    
    gbuffer_subpass.init_graph(frame_resources, frame_state, render_graph, gpu_device(), desc_alloc);
    backbuffer_subpass.init_graph(frame_resources, frame_state, desc_alloc);
  }
  
  void run() {
    bool quit = false; 

    while (!quit) {
      backbuffer_subpass.new_frame();
      
      ImGui::Begin("Settings");
      ImGui::Text("Hello world!");
      ImGui::End();


      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        frame_state.process_event(event);
        backbuffer_subpass.process_event(event);
        
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
    VkQueue queue = gpu_device().api_queue();
    VkSwapchainKHR swapchain = gpu_swapchain().api_swapchain();
    VkFence cmd_fence = submit_fences[frame_index];

    render_graph.build_barriers();
    //render_graph.dump_barriers();

    uint32_t image_index = 0;
    VKCHECK(vkAcquireNextImageKHR(device, swapchain, ~0ull, image_acquire_semaphores[frame_index], nullptr, &image_index)); 

    gpu::Image &image = backbuffers()[image_index];

    vkWaitForFences(device, 1, &cmd_fence, VK_TRUE, ~0ull);
    submit_fences[frame_index].reset();
    vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    desc_alloc.flip();

    auto ticks2 = SDL_GetTicks();
    frame_state.update(frame_index, image_index, (ticks2 - ticks)/1000.f);
    ticks = ticks2;

    render_graph.set_api_image(frame_resources.backbuffer, image.get_image());
    backbuffer_subpass.update_graph(frame_state);

    VkCommandBufferBeginInfo begin_cmd {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin_cmd);
    render_graph.write_commands(cmd);
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
  gpu::CmdBufferPool cmdbuffer_pool;

  const uint32_t frames_count = 0;
  uint32_t frame_index = 0;

  std::vector<VkCommandBuffer> cmd_buffers;
  std::vector<gpu::Fence> submit_fences;
  std::vector<gpu::Semaphore> image_acquire_semaphores;
  std::vector<gpu::Semaphore> submit_done_semaphores;  

  framegraph::RenderGraph render_graph;
  FrameResources frame_resources;
  FrameGlobal frame_state;
  GbufferSubpass gbuffer_subpass;
  BackbufferSubpass backbuffer_subpass;

  uint32_t ticks = 0;
};

RES_IMAGE_ID(GbufferAlbedo);
RES_IMAGE_ID(GbufferNormal);
RES_IMAGE_ID(GbufferMaterial);
RES_IMAGE_ID(GbufferDepth);

struct CBData {
  std::string name;
};

struct RGApp : SDLVulkanAppBase {
  RGApp(uint32_t w, uint32_t h) : SDLVulkanAppBase {w, h}, render_graph {gpu_device(), gpu_swapchain()} {}

  void init() {
    auto [width, height] = swapchain_fmt().extent2D();

    render_graph.add_task<CBData>("Gbuffer", 
      [&](CBData &data, rendergraph::RenderGraphBuilder &builder) {
        rendergraph::ImageDescriptor desc {
          VK_IMAGE_TYPE_2D,
          VK_FORMAT_R8G8B8A8_SRGB,
          VK_IMAGE_ASPECT_COLOR_BIT,
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
          width, height, 1, 1, 1
        };
        
        builder.create_image<GbufferAlbedo>(desc);
        builder.create_image<GbufferNormal>(desc);
        builder.create_image<GbufferMaterial>(desc);

        desc.format = VK_FORMAT_D16_UNORM;
        desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;

        builder.create_image<GbufferDepth>(desc);

        builder.use_color_attachment<GbufferAlbedo>();
        builder.use_color_attachment<GbufferNormal>();
        builder.use_color_attachment<GbufferMaterial>();
        builder.use_depth_attachment<GbufferDepth>();

        data.name = "Gbuffer";
      },
      [=](CBData &data, rendergraph::RenderResources &resources, VkCommandBuffer cmd) {
        std::cout << "In " << data.name << "\n";
      });

    render_graph.add_task<CBData>("Backbuffer", 
      [&](CBData &data, rendergraph::RenderGraphBuilder &builder){
        builder.sample_image<GbufferAlbedo>(VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.sample_image<GbufferNormal>(VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.sample_image<GbufferDepth>(VK_SHADER_STAGE_FRAGMENT_BIT);
        builder.sample_image<GbufferMaterial>(VK_SHADER_STAGE_FRAGMENT_BIT);
        data.name = "Backbuffer";
      },
      [=](CBData &data, rendergraph::RenderResources &resources, VkCommandBuffer cmd) {
        std::cout << "In " << data.name << "\n";
      });
    
    render_graph.add_task<CBData>("Somename",
      [&](CBData &data, rendergraph::RenderGraphBuilder &builder){
        builder.prepare_backbuffer();
        data.name = "PreparePresent";
      },
      [=](CBData &data, rendergraph::RenderResources &resources, VkCommandBuffer cmd) {
        std::cout << "In " << data.name << "\n";
      });
  }

  void submit() {
    render_graph.submit();
  }

private:
  rendergraph::RenderGraph render_graph;
}; 

int main() {
  RGApp app {800, 600};
  app.init();
  
  bool quit = false;
  while (!quit) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        quit = true;
      }
    } 
    app.submit(); 
  }
  return 0;
}
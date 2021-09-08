#ifndef IMGUI_CONTEXT_HPP_INCLUDED
#define IMGUI_CONTEXT_HPP_INCLUDED

#include <iostream>

#include <lib/imgui/imgui.h>
#include <lib/imgui/imgui_impl_sdl.h>
#include <lib/imgui/imgui_impl_vulkan.h>

#include "driver.hpp"

namespace gpu {

struct ImguiContext {
  ImguiContext(SDL_Window *sdl_window, const gpu::Instance &instance, const gpu::Device &device, uint32_t image_count, const gpu::RenderSubpass &subpass)
    : window {sdl_window}, pool {device.api_device(), 1}
  {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForVulkan(window);
    //ImGui_ImplVulkan_LoadFunctions
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance.api_instance();
    init_info.PhysicalDevice = device.api_physical_device();
    init_info.Device = device.api_device();
    init_info.QueueFamily = device.get_queue_family();
    init_info.Queue = device.api_queue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = pool.current_pool();
    init_info.Allocator = nullptr;
    init_info.MinImageCount = image_count;
    init_info.ImageCount = image_count;
    init_info.CheckVkResultFn = check_vk_result;
    init_info.Subpass = 0;
    ImGui_ImplVulkan_Init(&init_info, subpass.api_renderpass());
  }

  void new_frame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();
  }

  void render(VkCommandBuffer cmd) {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    
    ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
  }

  void process_event(const SDL_Event &event) {
    ImGui_ImplSDL2_ProcessEvent(&event);
  }

  void create_fonts(gpu::Device &device, VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo begin_cmd {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin_cmd);
    ImGui_ImplVulkan_CreateFontsTexture(cmd);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = 0,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr
    };

    auto fence = device.new_fence();
    VkFence api_fence = fence;
    VKCHECK(vkQueueSubmit(device.api_queue(), 1, &submit_info, fence));
    VKCHECK(vkWaitForFences(device.api_device(), 1, &api_fence, VK_TRUE, ~0ull));
  }

private:
  SDL_Window *window {nullptr};
  gpu::DescriptorPool pool;

  static void check_vk_result(VkResult err)
  {
    if (err == 0)
      return;
    std::cerr <<  "[vulkan] Error: VkResult = " <<  uint32_t(err) << "\n";
    if (err < 0)
      throw std::runtime_error {"Vulkan error"};
  }
};

}

#endif
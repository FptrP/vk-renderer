#include "gpu.hpp"

#include <optional>
#include <memory>

namespace gpu {
  
  static std::optional<Swapchain> g_swapchain;
  static std::unique_ptr<PipelinePool> g_pipeline_pool;
  static std::optional<SamplerPool> g_sampler_pool;
  
  void init_all(const InstanceConfig &icfg, PFN_vkDebugUtilsMessengerCallbackEXT callback, DeviceConfig dcfg, VkExtent2D window_size, SurfaceCreateCB &&surface_cb) {
    create_context(icfg, callback, dcfg, std::move(surface_cb));
    
    g_swapchain.emplace(Swapchain {
      window_size,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT});

    g_pipeline_pool.reset(new PipelinePool {});
    g_sampler_pool.emplace(SamplerPool {});
  }

  void close() {
    vkDeviceWaitIdle(app_device().api_device());

    auto ptr = g_pipeline_pool.get();
    delete ptr;

    g_pipeline_pool.release();
    g_sampler_pool.reset();
    g_swapchain.reset();
    close_context();
  }

  Swapchain &app_swapchain() {
    return g_swapchain.value();
  }
  
  PipelinePool &app_pipelines() {
    return *g_pipeline_pool;
  }

  GraphicsPipeline create_graphics_pipeline() {
    return {g_pipeline_pool.get()};
  }

  ComputePipeline create_compute_pipeline() {
    return {g_pipeline_pool.get()};
  }

  ComputePipeline create_compute_pipeline(const char *name) {
    gpu::ComputePipeline pipeline {g_pipeline_pool.get()};
    pipeline.set_program(name);
    return pipeline;
  }

  VkSampler create_sampler(const VkSamplerCreateInfo &info) {
    return g_sampler_pool->get_sampler(info);
  }

  void create_program(const std::string &name, std::initializer_list<ShaderBinding> shaders) {
    g_pipeline_pool->create_program(name, shaders);
  }

  void create_program(const std::string &name, std::vector<ShaderBinding> &&shaders) {
    g_pipeline_pool->create_program(name, std::move(shaders));
  }
  
  std::vector<CmdContext> allocate_cmd_contexts(CmdBufferPool &pool, uint32_t count) {
    auto &dev = app_device();
    auto api_buffers = pool.allocate(count);
    std::vector<CmdContext> cmd;
    cmd.reserve(count);
    for (auto elem : api_buffers) {
      cmd.emplace_back(dev.api_device(), elem, dev.get_allocator(), dev.get_properties().limits.minUniformBufferOffsetAlignment);
    }
    return cmd;
  }
  
  DescriptorPool new_descriptor_pool(uint32_t flips_count) {
    return DescriptorPool {flips_count};
  }
  
  CmdBufferPool new_command_pool() {
    return CmdBufferPool {};
  }
  
  Semaphore new_semaphore() {
    return Semaphore {};
  }
  
  Fence new_fence(bool signaled) {
    return Fence {signaled};
  }

  std::vector<Image> get_swapchain_images() {
    auto &ctx_dev = app_device();
    auto &ctx_swapchain = app_swapchain();
    auto images_count = get_swapchain_image_count();

    std::vector<VkImage> api_images;
    api_images.resize(images_count);
    VKCHECK(vkGetSwapchainImagesKHR(ctx_dev.api_device(), ctx_swapchain.api_swapchain(), &images_count, api_images.data()));

    std::vector<Image> images;
    images.reserve(images_count);
    for (auto handle : api_images) {
      images.emplace_back();
      images.back().create_reference(handle, ctx_swapchain.get_image_info());
    }
    return images;
  }
  
  uint32_t get_swapchain_image_count() {
    auto &swapchain = app_swapchain();
    return swapchain.get_images_count();
  }

}
#ifndef RENDERGRAPH_CPU_CTX_HPP_INCLUDED
#define RENDERGRAPH_CPU_CTX_HPP_INCLUDED

#include "gpu/driver.hpp"

#include "resources.hpp"

namespace rendergraph {

  struct GpuState {
    GpuState(gpu::Device &dev, gpu::Swapchain &swp)
      : device {dev}, 
        swapchain {swp},
        backbuffers_count { swapchain.get_images_count()},
        frames_count {backbuffers_count},
        cmdbuffer_pool {device.new_command_pool()},
        desc_pool {device.new_descriptor_pool(frames_count)},
        event_pool {device.api_device(), frames_count},
        cmd_buffers {device.allocate_cmd_contexts(cmdbuffer_pool, frames_count)}
    {
      submit_fences.reserve(frames_count);
      image_acquire_semaphores.reserve(frames_count);
      submit_done_semaphores.reserve(frames_count);

      for (uint32_t i = 0; i < frames_count; i++) {
        submit_fences.push_back(device.new_fence(true));
      }

      for (uint32_t i = 0; i < backbuffers_count; i++) {
        image_acquire_semaphores.push_back(device.new_semaphore());
        submit_done_semaphores.push_back(device.new_semaphore());
      }
    }

    ~GpuState() { vkDeviceWaitIdle(device.api_device()); }

    void acquire_image();
    void begin();
    void submit(bool present);

    gpu::CmdContext &get_cmdbuff() { return cmd_buffers[frame_index]; }
    
    uint32_t get_frame_index() const { return frame_index; }
    uint32_t get_backbuf_index() const { return backbuf_index; }

    std::vector<gpu::Image> take_backbuffers() { return device.get_swapchain_images(swapchain); }

    gpu::Device &get_device() { return device; }
    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) { return desc_pool.allocate_set(layout); }
    VkEvent allocate_event() { return event_pool.allocate(); }
    
    uint32_t get_frames_count() const { return frames_count; }
    uint32_t get_backbuffers_count() const { return backbuffers_count;}
  private:
    gpu::Device &device;
    gpu::Swapchain &swapchain;

    uint32_t backbuffers_count = 0;
    uint32_t frames_count = 0;

    gpu::CmdBufferPool cmdbuffer_pool;
    gpu::DescriptorPool desc_pool;
    gpu::EventPool event_pool;

    std::vector<gpu::CmdContext> cmd_buffers;
    std::vector<gpu::Fence> submit_fences;
    std::vector<gpu::Semaphore> image_acquire_semaphores;
    std::vector<gpu::Semaphore> submit_done_semaphores;  

    
    uint32_t frame_index = 0;
    uint32_t backbuf_index = 0;
    uint32_t backbuf_sem_index = 0;
  };

}

#endif
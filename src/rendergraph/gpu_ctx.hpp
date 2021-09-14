#ifndef RENDERGRAPH_CPU_CTX_HPP_INCLUDED
#define RENDERGRAPH_CPU_CTX_HPP_INCLUDED

#include "gpu/driver.hpp"

#include "resources.hpp"

namespace rendergraph {

  /*struct GpuContext {
    
    gpu::Image create_image(const ImageDescriptor &desc);
    gpu::Buffer create_buffer(const BufferDescriptor &desc);

    template <typename T>
    gpu::DynBuffer<T> create_dynbuffer(); 

    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) { return pool.allocate_set(layout); }

  private:
    gpu::Device &device;
    gpu::DescriptorPool &pool;
    
    uint32_t frames_count;
    uint32_t backbuffers_count;
    uint32_t frame_index;
    uint32_t backbuffer_index;
  };*/

  struct GpuState {
    GpuState(gpu::Device &dev, gpu::Swapchain &swp)
      : device {dev}, 
        swapchain {swp},
        vk_backbuffers {device.get_swapchain_images(swapchain)},
        frames_count {(uint32_t)vk_backbuffers.size()},
        cmdbuffer_pool {device.new_command_pool()},
        desc_pool {device.new_descriptor_pool(frames_count)},
        cmd_buffers {cmdbuffer_pool.allocate(frames_count)}
    {
      submit_fences.reserve(frames_count);
      image_acquire_semaphores.reserve(frames_count);
      submit_done_semaphores.reserve(frames_count);

      for (uint32_t i = 0; i < frames_count; i++) {
        submit_fences.push_back(device.new_fence(true));
        image_acquire_semaphores.push_back(device.new_semaphore());
        submit_done_semaphores.push_back(device.new_semaphore());
      }
    }

    ~GpuState() { vkDeviceWaitIdle(device.api_device()); }

    void begin();
    void submit();

    VkCommandBuffer get_cmdbuff() const { return cmd_buffers[frame_index]; }
    
    uint32_t get_frame_index() const { return frame_index; }
    uint32_t get_backbuf_index() const { return backbuf_index; }

    std::vector<gpu::Image> take_backbuffers() { return device.get_swapchain_images(swapchain); }

    gpu::Device &get_device() { return device; }
    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) { return desc_pool.allocate_set(layout); }

  private:
    gpu::Device &device;
    gpu::Swapchain &swapchain;

    std::vector<gpu::Image> vk_backbuffers;
    uint32_t frames_count = 0;

    gpu::CmdBufferPool cmdbuffer_pool;
    gpu::DescriptorPool desc_pool;

    std::vector<VkCommandBuffer> cmd_buffers;
    std::vector<gpu::Fence> submit_fences;
    std::vector<gpu::Semaphore> image_acquire_semaphores;
    std::vector<gpu::Semaphore> submit_done_semaphores;  

    
    uint32_t frame_index = 0;
    uint32_t backbuf_index = 0;
  };

}

#endif
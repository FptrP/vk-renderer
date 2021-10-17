#ifndef SYNC_PRIMIRIVE_HPP_INCLUDED
#define SYNC_PRIMIRIVE_HPP_INCLUDED

#include <algorithm>

#include "driver.hpp"
#include "resources.hpp"

namespace gpu {

  struct Fence {
    Fence(bool signaled = false) {

      VkFenceCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = signaled? VK_FENCE_CREATE_SIGNALED_BIT : 0u
      };

      VKCHECK(vkCreateFence(internal::app_vk_device(), &info, nullptr, &handle));
    }

    ~Fence() {
      if (handle) { vkDestroyFence(internal::app_vk_device(), handle, nullptr); }
    }

    Fence(Fence &&f) : handle {f.handle} {
      f.handle = nullptr;
    }

    const Fence &operator=(Fence &&o) {
      std::swap(handle, o.handle);
      return *this;
    }

    void reset() { vkResetFences(internal::app_vk_device(), 1, &handle); }
    
    operator VkFence() const { return handle; }

  private:
    VkFence handle {nullptr};

    Fence(const Fence&) = delete;
    const Fence &operator=(const Fence&) = delete;
  };

  struct Semaphore {
    Semaphore() {

      VkSemaphoreCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0
      };

      VKCHECK(vkCreateSemaphore(internal::app_vk_device(), &info, nullptr, &handle));
    }

    ~Semaphore() {
      if (handle) { vkDestroySemaphore(internal::app_vk_device(), handle, nullptr); }
    }

    Semaphore(Semaphore &&f) : handle {f.handle} {
      f.handle = nullptr;
    }

    const Semaphore &operator=(Semaphore &&o) {
      std::swap(handle, o.handle);
      return *this;
    }

    operator VkSemaphore() const { return handle; }

  private:
    VkSemaphore handle {nullptr};

    Semaphore(const Semaphore&) = delete;
    const Semaphore &operator=(const Semaphore&) = delete;
  };
  
  struct Barrier;

  struct ImageBarrier {
    ImageBarrier(const gpu::Image &img, VkImageAspectFlags aspect, VkAccessFlags src_access, VkAccessFlags dst_access, VkImageLayout src, VkImageLayout dst)
    {
      api_barrier = VkImageMemoryBarrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = src,
        .newLayout = dst,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img.get_image(),
        .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}
      };
    }

  private:
    VkImageMemoryBarrier api_barrier;

    friend Barrier;
  };

  struct Barrier {
    
    Barrier(VkPipelineStageFlags src_flags, VkPipelineStageFlags dst_flags, std::initializer_list<ImageBarrier> image)
      : src {src_flags}, dst {dst_flags}
    {
      for (auto b : image) {
        image_barriers.push_back(b.api_barrier);
      }
    }


    void flush(VkCommandBuffer cmd) const {
      vkCmdPipelineBarrier(cmd, src, dst, 0, memory_barriers.size(), memory_barriers.data(), 0, nullptr, image_barriers.size(), image_barriers.data());
    }

  private:
    VkPipelineStageFlags src = 0, dst = 0;
    std::vector<VkImageMemoryBarrier> image_barriers;
    std::vector<VkMemoryBarrier> memory_barriers;
  };

}

#endif
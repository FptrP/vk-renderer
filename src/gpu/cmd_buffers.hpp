#ifndef BUFFER_POOL_HPP_INCLUDED
#define BUFFER_POOL_HPP_INCLUDED

#include <vector>

#include "vkerror.hpp"

#include <vector>
#include <algorithm>

namespace gpu {


  struct CmdBufferPool {
    CmdBufferPool(VkDevice logical_device, uint32_t queue_family)
      : device {logical_device}
    {
      VkCommandPoolCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family
      };

      VKCHECK(vkCreateCommandPool(logical_device, &info, nullptr, &pool));
    }
    
    ~CmdBufferPool() {
      if (device && pool) {
        vkDestroyCommandPool(device, pool, nullptr);
      }
    }

    CmdBufferPool(CmdBufferPool &&o) : device {o.device}, pool {o.pool} { o.pool = nullptr; }
    
    const CmdBufferPool &operator=(CmdBufferPool &&o) {
      std::swap(device, o.device);
      std::swap(pool, o.pool);
      return *this;
    }

    std::vector<VkCommandBuffer> allocate(uint32_t count) {
      VkCommandBufferAllocateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count
      };

      std::vector<VkCommandBuffer> result;
      result.resize(count);
      VKCHECK(vkAllocateCommandBuffers(device, &info, result.data()));
      return result;
    }

    VkCommandBuffer allocate() {
      VkCommandBufferAllocateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
      };

      VkCommandBuffer result;
      VKCHECK(vkAllocateCommandBuffers(device, &info, &result));
      return result;
    }

  private:
    VkDevice device;
    VkCommandPool pool {VK_NULL_HANDLE};

    CmdBufferPool(const CmdBufferPool&) = delete;
    const CmdBufferPool &operator=(const CmdBufferPool&) = delete; 
  };

}

#endif
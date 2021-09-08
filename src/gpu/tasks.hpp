#ifndef TASKS_HPP_INCLUDED
#define TASKS_HPP_INCLUDED

#include "vkerror.hpp"

#include <functional>
#include <vector>

namespace gpu {

  using Task = std::function<void (VkCommandBuffer)>;

  struct TaskPool {
    TaskPool(VkDevice device, VkQueue queue, uint32_t queue_family);
    ~TaskPool();

    void run_task(Task task);
    void wait_all();

  private:
    VkDevice logical_device {nullptr};
    VkQueue device_queue {nullptr};
    uint32_t queue_family_index;

    VkCommandPool buffer_pool {nullptr};

    //std::vector<VkCommandBuffer> available_buffers;
    std::vector<VkCommandBuffer> used_buffers;
    //std::vector<VkFence> available_fences;
    std::vector<VkFence> used_fences;
  };

}


#endif
#ifndef FRAMEGRAPH_HPP_INCLUDED
#define FRAMEGRAPH_HPP_INCLUDED

#include "gpu/driver.hpp"

#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <optional>
#include <map>
#include <string>

namespace framegraph {

struct BufferDescriptor {
  uint32_t size;
  std::string name;
};

struct ImageDescriptor {
  uint32_t mip_levels;
  uint32_t array_layers;
  VkImageAspectFlags aspect;
  std::string name;
  bool reset_to_undefined_layout = false;
};

struct ImageSubresource {
  ImageSubresource(uint32_t image_id) : id {image_id} {}
  ImageSubresource(uint32_t image_id, uint32_t image_mip, uint32_t image_layer) 
    : id {image_id}, layer {image_layer}, mip {image_mip} {} 

  uint32_t id;
  uint32_t layer = 0;
  uint32_t mip = 0;
};
}


namespace std {
  template<>
  struct less<framegraph::ImageSubresource> : std::binary_function<framegraph::ImageSubresource, framegraph::ImageSubresource, bool> {
    bool operator()(const framegraph::ImageSubresource &left, const framegraph::ImageSubresource &right) const {
      return (left.id < right.id)
          || (left.layer < right.layer)
          || (left.mip < right.mip);
    }
  };
}

namespace framegraph {

using TaskCB = std::function<void(VkCommandBuffer)>;

struct Task {
  struct BufferAccess {
    uint32_t buffer_id;
    VkPipelineStageFlags stages;
    VkAccessFlags mem_accesses;
  };

  struct ImageSubresourceAccess {
    uint32_t image_id;
    uint32_t mip_level;
    uint32_t array_layer;
    VkPipelineStageFlags stages;
    VkAccessFlags mem_accesses;
    VkImageLayout layout;
  };


  std::vector<BufferAccess> used_buffers;
  std::vector<ImageSubresourceAccess> used_images;
  std::string name;
  TaskCB cb = nullptr;
};

struct Barrier {
  VkPipelineStageFlags src = 0;
  VkPipelineStageFlags dst = 0;
  
  struct ImageBarrier {
    VkAccessFlags src_access = 0;
    VkAccessFlags dst_access = 0;
    VkImageLayout src_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout dst_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t image_id = 0;
    uint32_t mip_level = 0;
    uint32_t array_layer = 0;
  };

  std::vector<ImageBarrier> image_barriers;
  std::vector<VkMemoryBarrier> buffer_barriers;
  std::vector<uint32_t> buffer_ids;
};

struct BufferTrackingState {
  bool dirty = false;
  uint32_t barrier_id = 0;

  VkPipelineStageFlags prev_stages = 0;
  VkAccessFlags prev_access = 0;

  VkPipelineStageFlags current_stages = 0;
  VkAccessFlags current_access = 0;
};

struct ImageSubresourceTrackingState {
  bool dirty = false;
  uint32_t barrier_id = 0; 

  VkPipelineStageFlags prev_stages = 0;
  VkAccessFlags prev_access = 0;
  VkImageLayout prev_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkPipelineStageFlags current_stages = 0;
  VkAccessFlags current_access = 0;
  VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

template <typename Container, typename Value>
uint32_t push_c(Container &cont, Value &&val) {
  uint32_t index = cont.size();
  cont.push_back(std::forward<Value>(val));
  return index;
}

struct RenderGraph {
  uint32_t create_buffer_desc(uint32_t size, const std::string &name) { 
    return push_c(buffers, BufferDescriptor {size, name});
  }
  
  uint32_t create_image_desc(uint32_t mip_levels, uint32_t array_layers, VkImageAspectFlags aspect, const std::string &name, bool reset = false) { 
    api_images.push_back(nullptr);
    return push_c(images, ImageDescriptor {mip_levels, array_layers, aspect, name, reset});
  }
  
  uint32_t add_task(Task &&t) { return push_c(tasks, std::move(t)); }

  void build_barriers();
  void dump_barriers();

  void set_api_image(uint32_t image_id, VkImage image) { api_images[image_id] = image; }
  void set_callback(uint32_t task_id, TaskCB &&cb) { tasks[task_id].cb = std::move(cb); }

  void write_commands(VkCommandBuffer cmd);

  void reset_buffer_state(uint32_t buffer_id, VkPipelineStageFlags stages, VkAccessFlags access);
  void reset_image_state(uint32_t image_id, uint32_t mip, uint32_t layer, VkPipelineStageFlags stages, VkAccessFlags access, VkImageLayout layout);

  const ImageDescriptor &get_descriptor(uint32_t image_id) const { return images[image_id]; }

private:
  std::vector<BufferDescriptor> buffers;
  std::vector<ImageDescriptor> images;
  std::vector<Task> tasks;
  std::vector<Barrier> barriers;

  std::vector<BufferTrackingState> buffer_states;
  std::vector<std::unique_ptr<ImageSubresourceTrackingState[]>> image_states;

  std::vector<VkImage> api_images;

  void create_barrier(uint32_t id, uint32_t mip, uint32_t layer, ImageSubresourceTrackingState &state);
  void create_barrier(uint32_t id, BufferTrackingState &state);
  void dump_barrier(uint32_t barrier_id);
  void write_barrier(const Barrier &barrier, VkCommandBuffer cmd);
};

struct SubpassDescriptor {
  SubpassDescriptor(RenderGraph &render_graph, const std::string &pass_name) : graph{render_graph}, name {pass_name} {}

  SubpassDescriptor &sample_image(uint32_t image_id, VkShaderStageFlags stages);
  SubpassDescriptor &use_color_attachment(uint32_t image_id);
  SubpassDescriptor &use_depth_attachment(uint32_t image_id);
  //SubpassDescriptor &use_storage_image(uint32_t image_id, bool readonly);
  //SubpassDescriptor &use_storage_buffer(uint32_t buffer_id, bool readonly);

  uint32_t flush_task();

private:
  RenderGraph &graph;
  std::string name;
  std::map<ImageSubresource, Task::ImageSubresourceAccess> images;
  std::map<uint32_t, Task::BufferAccess> buffers;
};

}
#endif
#include "framegraph.hpp"
#include <iostream>

static bool is_ro_access(VkAccessFlags flags) {

  const auto read_msk =
    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|
    VK_ACCESS_INDEX_READ_BIT|
    VK_ACCESS_INDIRECT_COMMAND_READ_BIT|
    VK_ACCESS_INPUT_ATTACHMENT_READ_BIT|
    VK_ACCESS_MEMORY_READ_BIT|
    VK_ACCESS_SHADER_READ_BIT|
    VK_ACCESS_TRANSFER_READ_BIT|
    VK_ACCESS_UNIFORM_READ_BIT|
    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;

  return (flags & read_msk);
}

static bool is_rw_access(VkAccessFlags flags) {
  const auto rw_msk = 
    VK_ACCESS_SHADER_WRITE_BIT|
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|
    VK_ACCESS_TRANSFER_WRITE_BIT|
    VK_ACCESS_MEMORY_WRITE_BIT;
  
  return (flags & rw_msk);
}

static bool merge_states(ImageSubresourceTrackingState &state, const Task::ImageSubresourceAccess &access, uint32_t task_id) {
  const bool empty_state = !state.prev_stages && !state.current_stages;

  if (empty_state || !state.dirty) {
    state.current_access = access.mem_accesses;
    state.current_layout = access.layout;
    state.current_stages = access.stages;
    state.dirty = true;
    state.barrier_id = 0;
    return true;
  }


  if (state.current_layout != access.layout) {
    return false;
  }

  //ro -> ro - merge state
  if (is_ro_access(state.current_access) && is_ro_access(access.mem_accesses)) {
    state.current_access |= access.mem_accesses;
    state.current_stages |= access.stages;  
    return true;
  }

  //ro -> rw, rw -> ro, rw -> rw --> can't merge
  return false;
}

static bool merge_states(BufferTrackingState &state, const Task::BufferAccess &access, uint32_t task_id) {
  const bool empty_state = !state.prev_stages && !state.current_stages;
  if (empty_state) {
    state.current_access = access.mem_accesses;
    state.current_stages = access.stages;
    state.dirty = true;
    state.barrier_id = 0;
    return true;
  }

  if (is_ro_access(state.current_access) && is_ro_access(access.mem_accesses)) {
    state.current_access |= access.mem_accesses;
    state.current_stages |= access.stages;
    return true;
  }
  return false;
}

void RenderGraph::build_barriers() {
  barriers.clear();
  barriers.resize(tasks.size(), Barrier{});

  if (!buffer_states.size()) {
    buffer_states.resize(buffers.size(), BufferTrackingState {});
  }

  if (!image_states.size()) {
    image_states.reserve(images.size());

    for (const auto &desc : images) {
      image_states.emplace_back(new ImageSubresourceTrackingState[desc.array_layers * desc.mip_levels]{});
    }
  }

  for (uint32_t task_id = 0; task_id < tasks.size(); task_id++) {
    const auto &task = tasks[task_id];
    
    for (const auto &image_elem : task.used_images) {
      const auto &image_desc = images[image_elem.image_id];
      auto &state = image_states[image_elem.image_id][image_elem.array_layer * image_desc.mip_levels + image_elem.mip_level];
      
      if (merge_states(state, image_elem, task_id)) {
        continue;
      }

      create_barrier(image_elem.image_id, image_elem.mip_level, image_elem.array_layer, state);
      
      state.barrier_id = task_id;
      state.dirty = true;
      state.current_stages = image_elem.stages;
      state.current_access = image_elem.mem_accesses;
      state.current_layout = image_elem.layout; 
    }

    for (const auto &buffer_elem : task.used_buffers) {
      auto &state = buffer_states[buffer_elem.buffer_id];
      
      if (merge_states(state, buffer_elem, task_id)) {
        continue;
      }

      create_barrier(buffer_elem.buffer_id, state);
      state.barrier_id = task_id;
      state.dirty = true;
      state.current_stages = buffer_elem.stages;
      state.current_access = buffer_elem.mem_accesses;
    }

  }

  for (uint32_t image_id = 0; image_id < images.size(); image_id++) {
    const auto &image_desc = images[image_id];
    for (uint32_t layer = 0; layer < image_desc.array_layers; layer++) {
      for (uint32_t mip = 0; mip < image_desc.mip_levels; mip++) {
        auto &state = image_states[image_id][layer * image_desc.mip_levels + mip];
        if (state.dirty) {
          create_barrier(image_id, mip, layer, state);
        }

        if (image_desc.reset_to_undefined_layout) {
          state.prev_layout = state.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
      }
    }
  }

  for (uint32_t buffer_id = 0; buffer_id < buffer_states.size(); buffer_id++) {
    auto &buf_state = buffer_states[buffer_id];
    if (buf_state.dirty) {
      create_barrier(buffer_id, buf_state);
    }
  }

  api_images.resize(images.size(), nullptr);
  task_callbacks.resize(tasks.size());
}

void RenderGraph::create_barrier(uint32_t id, uint32_t mip, uint32_t layer, ImageSubresourceTrackingState &state) {
  if (state.prev_stages == 0) {
    state.prev_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  }

  auto &barrier = barriers[state.barrier_id];
  barrier.src |= state.prev_stages;
  barrier.dst |= state.current_stages;

  Barrier::ImageBarrier image_barrier {
    .src_access = state.prev_access,
    .dst_access = state.current_access,
    .src_layout = state.prev_layout,
    .dst_layout = state.current_layout,
    .image_id = id,
    .mip_level = mip,
    .array_layer = layer
  };

  barrier.image_barriers.push_back(std::move(image_barrier));

  state.prev_stages = state.current_stages;
  state.prev_access = state.current_access;
  state.prev_layout = state.current_layout;
  state.current_access = 0;
  state.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  state.current_stages = 0;
  state.dirty = false;
}

void RenderGraph::create_barrier(uint32_t id, BufferTrackingState &state) {
  if (state.prev_stages == 0) {
    state.prev_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  }

  auto &barrier = barriers[state.barrier_id];
  barrier.src |= state.prev_stages;
  barrier.dst |= state.current_stages;

  VkMemoryBarrier mem_barrier {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = state.prev_access,
    .dstAccessMask = state.current_access
  };

  barrier.buffer_barriers.push_back(mem_barrier);
  barrier.buffer_ids.push_back(id);

  state.prev_stages = state.current_stages;
  state.prev_access = state.current_access;
  state.current_access = 0;
  state.current_stages = 0;
  state.dirty = false;
}

void RenderGraph::dump_barriers() {
  for (uint32_t i = 0; i < barriers.size(); i++) {
    std::cout << "Before task " << tasks[i].name << "\n";
    dump_barrier(i);
  }
}

#define PRINT_FLAG(flag_name) if (flags & flag_name) { \
  if (!first) std::cout << "|"; \
  std::cout << #flag_name ; \
  first = false; \
} 

static void dump_stages(VkPipelineStageFlags flags) {
  if (!flags) {
    std::cout << "0";
    return;
  }

  bool first = true;
  PRINT_FLAG(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_TRANSFER_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_HOST_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
  PRINT_FLAG(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
}

static void dump_access(VkAccessFlags flags) {
  if (!flags) {
    std::cout << "0";
    return;
  }

  bool first = true;
  PRINT_FLAG(VK_ACCESS_INDIRECT_COMMAND_READ_BIT)
  PRINT_FLAG(VK_ACCESS_INDEX_READ_BIT)
  PRINT_FLAG(VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
  PRINT_FLAG(VK_ACCESS_UNIFORM_READ_BIT)
  PRINT_FLAG(VK_ACCESS_INPUT_ATTACHMENT_READ_BIT)
  PRINT_FLAG(VK_ACCESS_SHADER_READ_BIT)
  PRINT_FLAG(VK_ACCESS_SHADER_WRITE_BIT)
  PRINT_FLAG(VK_ACCESS_COLOR_ATTACHMENT_READ_BIT)
  PRINT_FLAG(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
  PRINT_FLAG(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
  PRINT_FLAG(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
  PRINT_FLAG(VK_ACCESS_TRANSFER_READ_BIT)
  PRINT_FLAG(VK_ACCESS_TRANSFER_WRITE_BIT)
  PRINT_FLAG(VK_ACCESS_HOST_READ_BIT)
  PRINT_FLAG(VK_ACCESS_HOST_WRITE_BIT)
  PRINT_FLAG(VK_ACCESS_MEMORY_READ_BIT)
  PRINT_FLAG(VK_ACCESS_MEMORY_WRITE_BIT)
}

static void dump_layout(VkImageLayout layout) {
  #define PRINT(x) case x: std::cout << #x; break

  switch (layout) {
    PRINT(VK_IMAGE_LAYOUT_UNDEFINED);
    PRINT(VK_IMAGE_LAYOUT_GENERAL);
    PRINT(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_PREINITIALIZED);
    PRINT(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL);
    PRINT(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    default:
      std::cout << "EXT layout";
  }
} 

void RenderGraph::dump_barrier(uint32_t barrier_id) {
  const auto &barrier = barriers[barrier_id];
  if (barrier.src == 0 && barrier.dst == 0) {
    std::cout << "Empty barrier\n";
    return;
  }

  std::cout << "src_stage: ";
  dump_stages(barrier.src);
  std::cout << "\n";

  std::cout << "dst_stage: ";
  dump_stages(barrier.dst);
  std::cout << "\n";

  for (const auto &img_barrier : barrier.image_barriers) {
    std::cout << " - Image barrier for " << images[img_barrier.image_id].name << "\n";
    std::cout << " --- src_access : "; dump_access(img_barrier.src_access); std::cout << "\n";
    std::cout << " --- dst_access : "; dump_access(img_barrier.dst_access); std::cout << "\n";
    std::cout << " --- src_layout : "; dump_layout(img_barrier.src_layout); std::cout << "\n";
    std::cout << " --- dst_layout : "; dump_layout(img_barrier.dst_layout); std::cout << "\n";
  }

  for (uint32_t i = 0; i < barrier.buffer_barriers.size(); i++) {
    const auto buffer_id = barrier.buffer_ids[i];
    const auto &buf_barrier = barrier.buffer_barriers[i];

    std::cout << " - Memory barrier for " << buffers[buffer_id].name << "\n";
    std::cout << " --- src_access : "; dump_access(buf_barrier.srcAccessMask); std::cout << "\n";
    std::cout << " --- dst_access : "; dump_access(buf_barrier.dstAccessMask); std::cout << "\n";
  }

}

void RenderGraph::write_commands(VkCommandBuffer cmd) {
  for (uint32_t i = 0; i < tasks.size(); i++) {
    write_barrier(barriers[i], cmd);
    task_callbacks[i](cmd);
  }
}

void RenderGraph::write_barrier(const Barrier &barrier, VkCommandBuffer cmd) {
  std::vector<VkImageMemoryBarrier> img_barriers;
  for (auto &ib : barrier.image_barriers) {
    if (!api_images[ib.image_id]) {
      throw std::runtime_error {"Api image not set"};
    }
    
    img_barriers.push_back(VkImageMemoryBarrier {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = ib.src_access,
      .dstAccessMask = ib.dst_access,
      .oldLayout = ib.src_layout,
      .newLayout = ib.dst_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = api_images[ib.image_id],
      .subresourceRange = {images[ib.image_id].aspect, ib.mip_level, 1, ib.array_layer, 1}
    });
  }

  vkCmdPipelineBarrier(cmd,
    barrier.src,
    barrier.dst,
    0,
    barrier.buffer_barriers.size(),
    barrier.buffer_barriers.data(),
    0,
    nullptr,
    img_barriers.size(),
    img_barriers.data());

}

void RenderGraph::reset_buffer_state(uint32_t buffer_id, VkPipelineStageFlags stages, VkAccessFlags access) {
  auto &state = buffer_states[buffer_id];
  state.dirty = false;
  state.barrier_id = 0;
  state.current_access = state.prev_access = access;
  state.prev_stages = state.current_stages = stages;
}

void RenderGraph::reset_image_state(uint32_t image_id, uint32_t mip, uint32_t layer, VkPipelineStageFlags stages, VkAccessFlags access, VkImageLayout layout) {
  const auto &desc = images[image_id];
  auto &state = image_states[image_id][layer * desc.mip_levels + mip];
  state.dirty = false;
  state.barrier_id = 0;
  state.current_access = state.prev_access = access;
  state.prev_stages = state.current_stages = stages;
  state.prev_layout = state.current_layout = layout;
}
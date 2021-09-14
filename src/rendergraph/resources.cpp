#include "resources.hpp"

#include <iostream>

namespace rendergraph {
  
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

  static bool is_write_access(VkAccessFlags flags) {
    const auto rw_msk = 
      VK_ACCESS_SHADER_WRITE_BIT|
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|
      VK_ACCESS_TRANSFER_WRITE_BIT|
      VK_ACCESS_MEMORY_WRITE_BIT;
  
    return (flags & rw_msk);
  }

  static bool merge_states(ImageTrackingState &state, const ImageSubresourceState &access) {
    if (state.dst.layout != access.layout) {
      return false;
    }

    if (is_ro_access(state.dst.access) && is_ro_access(access.access)) {
      state.dst.stages |= access.stages;
      state.dst.access |= access.access;
      return true;
    }
    return false;
  }


  void TrackingState::add_input(const ResourceInput &input) {
    for (auto [subres, state] : input.images) {
      if (!images.count(subres)) {
        auto &track = images[subres]; 
        track.barrier_id = index;
        track.acquire_barrier = true;
        track.dst = state;
        continue;
      }

      auto &track = images[subres];

      if (merge_states(track, state)) {
        continue;
      }

      if (barriers.size() <= track.barrier_id) {
        barriers.resize(track.barrier_id + 1);
      }

      ImageBarrierState image_barrier {};
      image_barrier.image_hash = subres.image_hash;
      image_barrier.mip = subres.mip;
      image_barrier.layer = subres.layer;
      image_barrier.acquire_barrier = track.acquire_barrier;
      image_barrier.src = track.src;
      image_barrier.dst = track.dst;

      barriers[track.barrier_id].image_barriers.push_back(image_barrier);

      track.acquire_barrier = false;
      track.barrier_id = index;
      track.src = track.dst;
      track.dst = state;
    }

    for (auto [buf_id, state] : input.buffers) {
      if (!buffers.count(buf_id)) {
        auto &track = buffers[buf_id];
        track.acquire_barrier = true;
        track.barrier_id = index;
        track.dst = state;
        continue;
      }

      auto &track = buffers[buf_id];

      if (is_ro_access(track.dst.access) && is_ro_access(state.access)) {
        track.dst.stages |= state.stages;
        track.dst.access |= state.access;
        continue;
      }

      if (barriers.size() <= track.barrier_id) {
        barriers.resize(track.barrier_id + 1);
      }

      BufferBarrierState buffer_barrier {};
      buffer_barrier.buffer_hash = buf_id;
      buffer_barrier.acquire_barrier = track.acquire_barrier;
      buffer_barrier.src = track.src;
      buffer_barrier.dst = track.dst;

      barriers[track.barrier_id].buffer_barriers.push_back(buffer_barrier);

      track.acquire_barrier = false;
      track.barrier_id = index;
      track.src = track.dst;
      track.dst = state;
    }

    index++;
  }

  
  void TrackingState::flush() {
    for (auto [subres, track] : images) {

      if (barriers.size() <= track.barrier_id) {
        barriers.resize(track.barrier_id + 1);
      }

      ImageBarrierState image_barrier {};
      image_barrier.image_hash = subres.image_hash;
      image_barrier.mip = subres.mip;
      image_barrier.layer = subres.layer;
      image_barrier.acquire_barrier = track.acquire_barrier;
      image_barrier.src = track.src;
      image_barrier.dst = track.dst;

      barriers[track.barrier_id].image_barriers.push_back(image_barrier);
      track.src = track.dst;
    }

    for (auto [buf_id, track] : buffers) {

      if (barriers.size() <= track.barrier_id) {
        barriers.resize(track.barrier_id + 1);
      }

      BufferBarrierState buffer_barrier {};
      buffer_barrier.buffer_hash = buf_id;
      buffer_barrier.acquire_barrier = track.acquire_barrier;
      buffer_barrier.src = track.src;
      buffer_barrier.dst = track.dst;

      barriers[track.barrier_id].buffer_barriers.push_back(buffer_barrier);
      track.src = track.dst;
    }
  }

  void TrackingState::clear() {
    index = 0;
    buffers.clear();
    images.clear();
    barriers.clear();
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

  void TrackingState::dump_barrier(uint32_t barrier_id) {
    const auto &barrier = barriers.at(barrier_id);

    for (const auto &img_barrier : barrier.image_barriers) {
      std::cout << " - Image barrier " << "\n";
      std::cout << " --- hash " << img_barrier.image_hash << "\n";
      std::cout << " --- mip = " << img_barrier.mip << " layer = " << img_barrier.layer << "\n";

      if (img_barrier.acquire_barrier) {
        std::cout << " --- acquire_barrier\n";
      } else {
        std::cout << " --- src_stages : "; dump_stages(img_barrier.src.stages); std::cout << "\n";
        std::cout << " --- src_access : "; dump_access(img_barrier.src.access); std::cout << "\n";
        std::cout << " --- src_layout : "; dump_layout(img_barrier.src.layout); std::cout << "\n";
      }
      
      std::cout << " --- dst_stages : "; dump_stages(img_barrier.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(img_barrier.dst.access); std::cout << "\n";
      std::cout << " --- dst_layout : "; dump_layout(img_barrier.dst.layout); std::cout << "\n";
    }

    for (const auto &buf_barrier : barrier.buffer_barriers) {

      std::cout << " - Memory barrier for buffer " << buf_barrier.buffer_hash << "\n";
      if (buf_barrier.acquire_barrier) {
        std::cout << " --- acquire_barrier\n";
      } else {
        std::cout << " --- src_stages : "; dump_stages(buf_barrier.src.stages); std::cout << "\n";
        std::cout << " --- src_access : "; dump_access(buf_barrier.src.access); std::cout << "\n";
      }
      
      std::cout << " --- dst_stages : "; dump_stages(buf_barrier.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(buf_barrier.dst.access); std::cout << "\n";
    }
  }

  void TrackingState::dump_barriers() { 
    for (uint32_t i = 0; i < barriers.size(); i++) {
      std::cout << "Barrier " << i << "\n";
      dump_barrier(i);
    }
  } 

}
#include "resources.hpp"

#include <iostream>

namespace rendergraph {
  
  ImageResourceId GraphResources::create_global_image(const ImageDescriptor &desc) {
    uint32_t remap_index = image_remap.size();
    uint32_t image_index = global_images.size();
    
    uint32_t count = desc.array_layers * desc.mip_levels;
    std::unique_ptr<ImageSubresourceState[]> ptr;
    ptr.reset(new ImageSubresourceState[count]);

    global_images.emplace_back(GlobalImage {
      gpu::Image {api_device, allocator}, 
      std::move(ptr)
    });

    global_images.back().vk_image.create(desc.type, desc.get_vk_info(), desc.tiling, desc.usage);
    image_remap.emplace_back(image_index);
    
    ImageResourceId id {};
    id.index = remap_index;
    return id;
  }

  ImageResourceId GraphResources::create_global_image_ref(gpu::Image &image) {
    uint32_t remap_index = image_remap.size();
    uint32_t image_index = global_images.size();
    const auto &desc = image.get_info();

    uint32_t count = desc.array_layers * desc.mip_levels;
    std::unique_ptr<ImageSubresourceState[]> ptr;
    ptr.reset(new ImageSubresourceState[count]);

    global_images.emplace_back(GlobalImage {
      gpu::Image {api_device, allocator}, 
      std::move(ptr)
    });

    global_images.back().vk_image.create_reference(image.get_image(), desc);
    image_remap.emplace_back(image_index);
    
    ImageResourceId id {};
    id.index = remap_index;
    return id;
  }
  
  BufferResourceId GraphResources::create_global_buffer(const BufferDescriptor &desc) {
    uint32_t remap_index = buffer_remap.size();
    uint32_t buffer_index = global_buffers.size();

    global_buffers.emplace_back(GlobalBuffer {
      gpu::Buffer {allocator},
      {}
    });

    global_buffers.back().vk_buffer.create(desc.memory_type, desc.size, desc.usage);
    buffer_remap.emplace_back(buffer_index);

    BufferResourceId id {};
    id.index = remap_index;
    return id;
  }
  
  void GraphResources::remap(ImageResourceId src, ImageResourceId dst) {
    std::swap(image_remap.at(src.index), image_remap.at(dst.index));
  }
  
  void GraphResources::remap(BufferResourceId src, BufferResourceId dst) {
    std::swap(buffer_remap.at(src.index), buffer_remap.at(dst.index));
  }

  const gpu::ImageInfo &GraphResources::get_info(ImageResourceId id) const {
    auto index = image_remap.at(id.index);
    return global_images.at(index).vk_image.get_info(); 
  }

  gpu::Image &GraphResources::get_image(ImageResourceId id) {
    auto index = image_remap.at(id.index);
    return global_images.at(index).vk_image;
  }
  
  gpu::Buffer &GraphResources::get_buffer(BufferResourceId id) {
    auto index = buffer_remap.at(id.index);
    return global_buffers.at(index).vk_buffer;
  }

  void GraphResources::set_input_state(BufferResourceId id, BufferState state) {
    auto index = buffer_remap.at(id.index);
    auto &desc = global_buffers.at(index);
    desc.input_state = state;
  } 
  
  void GraphResources::set_input_state(ImageSubresourceId id, ImageSubresourceState state) {
    auto index = image_remap.at(id.id.index);
    auto &desc = global_images.at(index);
    auto &info = desc.vk_image.get_info();
    uint32_t offset = id.layer * info.mip_levels + id.mip;
    desc.input_state[offset] = state;
  }

  const BufferState &GraphResources::get_input_state(BufferResourceId id) const {
    auto index = buffer_remap.at(id.index);
    auto &desc = global_buffers.at(index);
    return desc.input_state;
  }
  
  const ImageSubresourceState &GraphResources::get_input_state(ImageSubresourceId id) const {
    auto index = image_remap.at(id.id.index);
    auto &desc = global_images.at(index);
    auto &info = desc.vk_image.get_info();
    uint32_t offset = id.layer * info.mip_levels + id.mip;
    return desc.input_state[offset];
  }

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


  void TrackingState::add_input(const ResourceInput &input, const GraphResources &resources) {
    for (auto [subres, state] : input.images) {
      if (!images.count(subres)) {
        auto &track = images[subres]; 
        track.barrier_id = index;
        track.src = resources.get_input_state(subres);
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
      image_barrier.id = subres;
      image_barrier.src = track.src;
      image_barrier.dst = track.dst;

      barriers[track.barrier_id].image_barriers.push_back(image_barrier);

      track.barrier_id = index;
      track.src = track.dst;
      track.dst = state;
    }

    for (auto [buf_id, state] : input.buffers) {
      if (!buffers.count(buf_id)) {
        auto &track = buffers[buf_id];
        track.barrier_id = index;
        track.src = resources.get_input_state(buf_id);
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
      buffer_barrier.id = buf_id;
      buffer_barrier.src = track.src;
      buffer_barrier.dst = track.dst;

      barriers[track.barrier_id].buffer_barriers.push_back(buffer_barrier);

      track.barrier_id = index;
      track.src = track.dst;
      track.dst = state;
    }
    index++;
  }

  
  void TrackingState::flush(GraphResources &resources) {
    for (auto [subres, track] : images) {

      if (barriers.size() <= track.barrier_id) {
        barriers.resize(track.barrier_id + 1);
      }

      ImageBarrierState image_barrier {};
      image_barrier.id = subres;
      image_barrier.src = track.src;
      image_barrier.dst = track.dst;

      barriers[track.barrier_id].image_barriers.push_back(image_barrier);
      track.src = track.dst;
      resources.set_input_state(subres, track.dst);
    }

    for (auto [buf_id, track] : buffers) {

      if (barriers.size() <= track.barrier_id) {
        barriers.resize(track.barrier_id + 1);
      }

      BufferBarrierState buffer_barrier {};
      buffer_barrier.id = buf_id;
      buffer_barrier.src = track.src;
      buffer_barrier.dst = track.dst;

      barriers[track.barrier_id].buffer_barriers.push_back(buffer_barrier);
      track.src = track.dst;
      resources.set_input_state(buf_id, track.dst);
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
      std::cout << " --- id " << img_barrier.id.id.get_index() << "\n";
      std::cout << " --- mip = " << img_barrier.id.mip << " layer = " << img_barrier.id.layer << "\n";

      std::cout << " --- src_stages : "; dump_stages(img_barrier.src.stages); std::cout << "\n";
      std::cout << " --- src_access : "; dump_access(img_barrier.src.access); std::cout << "\n";
      std::cout << " --- src_layout : "; dump_layout(img_barrier.src.layout); std::cout << "\n";
      
      std::cout << " --- dst_stages : "; dump_stages(img_barrier.dst.stages); std::cout << "\n";
      std::cout << " --- dst_access : "; dump_access(img_barrier.dst.access); std::cout << "\n";
      std::cout << " --- dst_layout : "; dump_layout(img_barrier.dst.layout); std::cout << "\n";
    }

    for (const auto &buf_barrier : barrier.buffer_barriers) {

      std::cout << " - Memory barrier for buffer " << buf_barrier.id.get_index() << "\n";

      std::cout << " --- src_stages : "; dump_stages(buf_barrier.src.stages); std::cout << "\n";
      std::cout << " --- src_access : "; dump_access(buf_barrier.src.access); std::cout << "\n";
      
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
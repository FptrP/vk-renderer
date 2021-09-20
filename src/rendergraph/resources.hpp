#ifndef RENDERGRAPH_RESOURCES_HPP_INLUDED
#define RENDERGRAPH_RESOURCES_HPP_INLUDED

#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <memory>
#include <functional>
#include <unordered_map>

#include "gpu/driver.hpp"

namespace rendergraph {

  struct ImageDescriptor {
    VkImageType type;
    VkFormat format;
    VkImageAspectFlags aspect = 0;
    VkImageTiling tiling;
    VkImageUsageFlags usage;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;

    gpu::ImageInfo get_vk_info() const {
      return {
        format,
        aspect,
        width,
        height,
        depth,
        mip_levels,
        array_layers
      };
    }
  };

  struct BufferDescriptor {
    uint64_t size;
    VkBufferUsageFlags usage;
    VmaMemoryUsage memory_type;
  };

  struct GraphResources;

  struct ImageResourceId {
    ImageResourceId() {}
    uint32_t get_index() const { return index; }

    bool operator==(const ImageResourceId &id) const { return index == id.index; }

  private:
    uint32_t index;
    friend struct GraphResources;
  };
  
  struct BufferResourceId {
    BufferResourceId() {}
    uint32_t get_index() const { return index; }

    bool operator==(const BufferResourceId &id) const { return index == id.index; }

  private:
    uint32_t index;
    friend struct GraphResources;
  };

  struct ImageSubresourceId {
    ImageResourceId id;
    uint32_t mip = 0;
    uint32_t layer = 0;

    bool operator==(const ImageSubresourceId &l) const { 
      return id == l.id && layer == l.layer && mip == l.mip; 
    }

  };

  struct ImageSubresourceHashFunc {
    template <typename T>
    static inline void hash_combine(std::size_t &s, const T &v) {
      std::hash<T> h;
      s ^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2); 
    }

    std::size_t operator()(const ImageSubresourceId &res) const {
      std::size_t h = 0;
      hash_combine(h, res.id.get_index());
      hash_combine(h, res.layer);
      hash_combine(h, res.mip);
      return h;
    }
  };

  struct BufferHashFunc {
    std::size_t operator()(const BufferResourceId &res) const {
      return res.get_index();
    }
  };

  struct ImageViewId {
    ImageViewId(ImageResourceId id, gpu::ImageViewRange view) : res_id {id}, range {view} {}
    ImageViewId() {}
    
    ImageResourceId get_id() const { return res_id; }
    const gpu::ImageViewRange &get_range() const { return range; }
    operator ImageResourceId() const { return res_id; } 
  private:
    ImageResourceId res_id;
    gpu::ImageViewRange range;
  };

  struct ImageSubresourceState {
    VkPipelineStageFlags stages = 0;
    VkAccessFlags access = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED; 
  };

  struct BufferState {
    VkPipelineStageFlags stages = 0;
    VkAccessFlags access = 0;
  };

  struct BufferBarrierState {
    BufferResourceId id;
    BufferState src;
    BufferState dst;
  };
  
  struct ImageBarrierState {
    ImageSubresourceId id; 
    ImageSubresourceState src;
    ImageSubresourceState dst;
  };

  constexpr uint32_t INVALID_BARRIER_INDEX = UINT32_MAX;

  struct ImageTrackingState {
    uint32_t barrier_id = INVALID_BARRIER_INDEX;
    ImageSubresourceState src;
    ImageSubresourceState dst;
  };

  struct BufferTrackingState {
    uint32_t barrier_id = INVALID_BARRIER_INDEX;
    BufferState src;
    BufferState dst;
  };

  struct Barrier {
    std::vector<BufferBarrierState> buffer_barriers;
    std::vector<ImageBarrierState> image_barriers;

    bool is_empty() const { return buffer_barriers.empty() && image_barriers.empty(); }
  };

  struct GraphResources {
    GraphResources(VmaAllocator alloc, VkDevice dev) : allocator {alloc}, api_device {dev} {}
    
    ImageResourceId create_global_image(const ImageDescriptor &desc);
    ImageResourceId create_global_image_ref(gpu::Image &image);

    BufferResourceId create_global_buffer(const BufferDescriptor &desc);

    void remap(ImageResourceId src, ImageResourceId dst);
    void remap(BufferResourceId src, BufferResourceId dst);

    const gpu::ImageInfo &get_info(ImageResourceId id) const;
    gpu::Image &get_image(ImageResourceId id);
    gpu::Buffer &get_buffer(BufferResourceId id);

    const BufferTrackingState &get_resource_state(BufferResourceId id) const;
    const ImageTrackingState &get_resource_state(ImageSubresourceId id) const;
    
    BufferTrackingState &get_resource_state(BufferResourceId id);
    ImageTrackingState &get_resource_state(ImageSubresourceId id);
  
  private:
    
    struct GlobalImage {
      gpu::Image vk_image;
      std::unique_ptr<ImageTrackingState[]> states;
    };
    
    struct GlobalBuffer {
      gpu::Buffer vk_buffer;
      BufferTrackingState state;
    };

    struct FrameImage {
      gpu::Image vk_image;
    };

    struct FrameBuffer {
      gpu::Buffer vk_buffer;
    };
    
    VmaAllocator allocator {nullptr};
    VkDevice api_device {nullptr};

    std::vector<uint32_t> image_remap;
    std::vector<uint32_t> buffer_remap;
    std::vector<GlobalImage> global_images;
    std::vector<GlobalBuffer> global_buffers;
  };

  struct TrackingState {
    void add_input(GraphResources &resources, const BufferResourceId &id, const BufferState &state);
    void add_input(GraphResources &resources, const ImageSubresourceId &id, const ImageSubresourceState &state);
    void next_task() { index++; }

    void flush(GraphResources &resources);
    void dump_barriers();
    void clear();

    const std::vector<Barrier> &get_barriers() { return barriers; }
    std::vector<Barrier> take_barriers() { return std::move(barriers); }
    
  private:
    uint32_t index = 0;
    std::vector<BufferResourceId> dirty_buffers;
    std::vector<ImageSubresourceId> dirty_images;
    std::vector<Barrier> barriers;

    void dump_barrier(uint32_t barrier_id);
  };

  

}


#endif
#ifndef RENDERGRAPH_RESOURCES_HPP_INLUDED
#define RENDERGRAPH_RESOURCES_HPP_INLUDED

#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <memory>
#include <functional>
#include <map>

#include "gpu/driver.hpp"

#define RES_IMAGE_ID(name) struct name : rendergraph::BaseImageID {}
#define RES_BUFFER_ID(name) struct name : rendergraph::BaseBufferID {}

namespace rendergraph {
  struct ImageSubresource {
    std::size_t image_hash;
    uint32_t layer = 0;
    uint32_t mip = 0;
  };
}

namespace std {
  template<>
  struct less<rendergraph::ImageSubresource> : std::binary_function<rendergraph::ImageSubresource, rendergraph::ImageSubresource, bool> {
    bool operator()(const rendergraph::ImageSubresource &left, const rendergraph::ImageSubresource &right) const {
      return (left.image_hash < right.image_hash)
          || (left.layer < right.layer)
          || (left.mip < right.mip);
    }
  };
}

namespace rendergraph {

  struct BaseImageID {};
  struct BaseBufferID {};

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
  
  struct ImageRef {
    ImageRef(std::size_t id, gpu::ImageViewRange view) : hash {id}, range {view} {}

    std::size_t get_hash() const { return hash; }
    const gpu::ImageViewRange &get_range() const { return range; }
  
  private:
    std::size_t hash;
    gpu::ImageViewRange range;
  };

  struct BufferDescriptor {
    uint64_t size;
    VkBufferUsageFlags usage;
    VmaMemoryUsage memory_type;
  };

  template <typename Image>
  std::size_t get_image_hash() {
    static_assert(std::is_base_of_v<BaseImageID, Image>, "ImageID expected");
    std::type_index index {typeid(Image)};
    return index.hash_code();
  }

  template <typename Buffer>
  std::size_t get_buffer_hash() {
    static_assert(std::is_base_of_v<BaseBufferID, Buffer>, "ImageID expected");
    std::type_index index {typeid(Buffer)};
    return index.hash_code();
  }

  std::size_t get_backbuffer_hash();

  struct ImageSubresourceState {
    VkPipelineStageFlags stages = 0;
    VkAccessFlags access = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED; 
  };

  struct BufferState {
    VkPipelineStageFlags stages = 0;
    VkAccessFlags access = 0;
  };

  struct Image {
    gpu::Image vk_image;
    std::unique_ptr<ImageSubresourceState[]> input_state;

    Image(gpu::Image &&img) : vk_image {std::move(img)} {
      const auto &desc = vk_image.get_info();
      input_state.reset(new ImageSubresourceState[desc.mip_levels * desc.array_layers]); 
    }

    ImageSubresourceState &get_external_state(ImageSubresource subres) {
      const auto &desc = vk_image.get_info();
      return input_state[desc.mip_levels * subres.layer + subres.mip];
    }

    const ImageSubresourceState &get_external_state(ImageSubresource subres) const {
      const auto &desc = vk_image.get_info();
      return input_state[desc.mip_levels * subres.layer + subres.mip];
    }

  };

  struct Buffer {
    gpu::Buffer vk_buffer;
    BufferState input_state;
  };
  
  struct BufferBarrierState {
    std::size_t buffer_hash;
    BufferState src;
    BufferState dst;
    bool acquire_barrier; //use input_state 
  };
  
  struct ImageBarrierState {
    std::size_t image_hash; //to ref img
    uint32_t mip;
    uint32_t layer;
    ImageSubresourceState src;
    ImageSubresourceState dst;
    bool acquire_barrier;
  };

  struct ImageTrackingState {
    uint32_t barrier_id;
    bool acquire_barrier;
    ImageSubresourceState src;
    ImageSubresourceState dst;
  };

  struct BufferTrackingState {
    uint32_t barrier_id;
    bool acquire_barrier;
    BufferState src;
    BufferState dst;
  };

  struct ResourceInput {
    std::map<std::size_t, BufferState> buffers;
    std::map<ImageSubresource, ImageSubresourceState> images;
  };

  struct Barrier {
    std::vector<BufferBarrierState> buffer_barriers;
    std::vector<ImageBarrierState> image_barriers;
    bool is_empty() const { return buffer_barriers.empty() && image_barriers.empty(); }
  };

  struct GraphResources {
    std::unordered_map<std::size_t, uint32_t> image_remap;
    std::unordered_map<std::size_t, uint32_t> buffer_remap;
    std::vector<Image> images;
    std::vector<Buffer> buffers;
  };

  struct TrackingState {
    void add_input(const ResourceInput &input);
    void flush();
    void dump_barriers();
    void clear();
    bool is_dirty() const { return dirty; }

    const std::vector<Barrier> &get_barriers() { return barriers; }
    
    void set_external_state(GraphResources &resources);

  private:
    bool dirty = false;
    uint32_t index = 0;
    std::map<std::size_t, BufferTrackingState> buffers;
    std::map<ImageSubresource, ImageTrackingState> images;
    std::vector<Barrier> barriers;

    void dump_barrier(uint32_t barrier_id);
  };

  

}


#endif
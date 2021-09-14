#ifndef DESCRIPTORS_HPP_INCLUDED
#define DESCRIPTORS_HPP_INCLUDED

#include "resources.hpp"
#include "dynbuffer.hpp"
#include "driver.hpp"

#include <unordered_map>

namespace gpu {

  struct DescriptorWriter {
    DescriptorWriter(VkDescriptorSet set) : target {set} {}

    template<typename T>
    DescriptorWriter &bind_dynbuffer(uint32_t binding, const DynBuffer<T> &buf) {
      buffers.push_back({binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, buf.api_buffer(), 0, sizeof(T)});
      return *this;
    }

    DescriptorWriter &bind_storage_buffer(uint32_t binding, const Buffer &buf) {
      buffers.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buf.get_api_buffer(), 0, buf.get_size()});
      return *this;
    }

    DescriptorWriter &bind_image(uint32_t binding, Image &image, const Sampler &sampler) {
      ImageViewRange range {};
      range.type = VK_IMAGE_VIEW_TYPE_2D;
      range.base_layer = 0;
      range.base_mip = 0;
      range.layers_count = 1;
      range.mips_count = image.get_mip_levels();

      auto view = image.get_view(range);
      images.push_back({binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, view, sampler.api_sampler()});
      return *this;
    }


    void write(VkDevice device) {
      std::vector<VkDescriptorBufferInfo> buffer_info;
      std::vector<VkWriteDescriptorSet> writes;
      
      buffer_info.reserve(buffers.size());
      writes.reserve(buffers.size());


      for (const auto &buf : buffers) {
        buffer_info.push_back({buf.buffer, buf.offset, buf.range});
      }

      for (uint32_t i = 0; i < buffers.size(); i++) {
        VkWriteDescriptorSet w {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = target;
        w.dstBinding = buffers[i].binding;
        w.descriptorType = buffers[i].type;
        w.descriptorCount = 1;
        w.pBufferInfo = &buffer_info[i];
        writes.push_back(w);
      }
      
      vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
      buffer_info.clear();
      writes.clear();

      std::vector<VkDescriptorImageInfo> image_info;
      for (const auto &img : images) {
        image_info.push_back({img.sampler, img.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
      }

      for (uint32_t i = 0; i < images.size(); i++) {
        VkWriteDescriptorSet w {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = target;
        w.dstBinding = images[i].binding;
        w.descriptorType = images[i].type;
        w.descriptorCount = 1;
        w.pImageInfo = &image_info[i];
        writes.push_back(w);
      }

      vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
    }

  private:
    struct BufferBinding;
    struct ImageBinding;
    
    VkDescriptorSet target;
    std::vector<BufferBinding> buffers;
    std::vector<ImageBinding> images;

    struct BufferBinding {
      uint32_t binding;
      VkDescriptorType type;
      VkBuffer buffer;
      uint64_t offset;
      uint64_t range;
    };

    struct ImageBinding {
      uint32_t binding;
      VkDescriptorType type;
      VkImageView image;
      VkSampler sampler;
    };

  };


}

#endif
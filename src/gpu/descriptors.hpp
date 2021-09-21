#ifndef DESCRIPTORS_HPP_INCLUDED
#define DESCRIPTORS_HPP_INCLUDED

#include "resources.hpp"
#include "dynbuffer.hpp"

namespace gpu {
  struct BaseBinding;

  namespace internal {
    inline void write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0);
  }

  struct BaseBinding {
    BaseBinding(uint32_t base_binding, uint32_t dst_array_elem, uint32_t desc_count, VkDescriptorType type) {
      desc_write.dstBinding = base_binding;
      desc_write.dstArrayElement = dst_array_elem;
      desc_write.descriptorCount = desc_count; 
      desc_write.descriptorType = type;
    }
  protected:
    VkWriteDescriptorSet desc_write { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };

    friend inline void internal::write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0);
  };

  
  struct DynBufBinding : BaseBinding {
    template <typename T>
    DynBufBinding(uint32_t binding, const DynBuffer<T> &buf) 
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}
    {
      info.buffer = buf.api_buffer();
      info.offset = 0;
      info.range = sizeof(T);

      desc_write.pBufferInfo = &info;
    }

  private:
    VkDescriptorBufferInfo info {};
  };

  struct TextureBinding : BaseBinding {
    TextureBinding(uint32_t binding, Image &image, VkSampler sampler, uint32_t base_mip, uint32_t mips_count)
      : BaseBinding {binding, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}
    {
      info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      info.sampler = sampler;

      ImageViewRange range {};
      range.type = VK_IMAGE_VIEW_TYPE_2D;
      range.base_layer = 0;
      range.base_mip = base_mip;
      range.layers_count = 1;
      range.mips_count = mips_count;

      info.imageView = image.get_view(range);
      desc_write.pImageInfo = &info;
    }

    TextureBinding(uint32_t binding, Image &image, VkSampler sampler) : TextureBinding {binding, image, sampler, 0, image.get_mip_levels()} {}
  private:
    VkDescriptorImageInfo info {};
  };

  namespace internal {
    inline void write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0) {
      *ptr = b0.desc_write;
      ptr->dstSet = set;
    }
  
    template <typename... Bindings>
    void write_set_base(VkDevice api_device, VkDescriptorSet set, VkWriteDescriptorSet *ptr, const BaseBinding &b0, const Bindings&... rest)
    {
      write_set_base(api_device, set, ptr, b0);
      write_set_base(api_device, set, ptr + 1, rest...);
    }

    template <typename... Bindings> 
    void write_set(VkDevice api_device, VkDescriptorSet set, const Bindings&... bindings) {
      constexpr auto count = sizeof...(bindings);

      VkWriteDescriptorSet writes[count];
      write_set_base(api_device, set, writes, bindings...);

      vkUpdateDescriptorSets(api_device, count, writes, 0, nullptr);
    }
  }

  
}

#endif
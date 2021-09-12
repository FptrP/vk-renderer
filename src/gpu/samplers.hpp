#ifndef SAMPLERS_HPP_INCLUDED
#define SAMPLERS_HPP_INCLUDED

#include "vkerror.hpp"

#include <algorithm>

namespace gpu {

  /*struct SamplersPool {
    
    void f() {
      VkSamplerCreateInfo o;
    }

  private:

  };*/

  struct Sampler {
    Sampler(VkDevice device, VkSamplerCreateInfo info) : dev {device} {
      info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      info.pNext = nullptr;
      VKCHECK(vkCreateSampler(device, &info, nullptr, &handle));
    }

    ~Sampler() {
      if (dev && handle)
        vkDestroySampler(dev, handle, nullptr);
    }

    VkSampler api_sampler() const { return handle; }

    Sampler(Sampler &&s) : dev {s.dev}, handle {s.handle} { s.handle = nullptr; }
    const Sampler &operator=(Sampler &&s) {
      std::swap(dev, s.dev);
      std::swap(handle, s.handle);
      return *this;
    }

  private:
    VkDevice dev {nullptr};
    VkSampler handle {nullptr};

    Sampler(const Sampler &) = delete;
    const Sampler &operator=(const Sampler &) = delete;
  };

  void bind_descriptors(VkCommandBuffer cmd, Pipeline &pipeline, uint32_t first_set, std::initializer_list<VkDescriptorSet> sets, std::initializer_list<uint32_t> offsets);
  void bind_descriptors(VkCommandBuffer cmd, Pipeline &pipeline, uint32_t first_set, std::initializer_list<VkDescriptorSet> sets);
}

#endif
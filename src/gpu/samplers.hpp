#ifndef SAMPLERS_HPP_INCLUDED
#define SAMPLERS_HPP_INCLUDED

#include "vkerror.hpp"

#include <algorithm>
#include <unordered_map>
#include <functional>

namespace gpu {

  struct SamplerHashFunc {
    std::size_t operator()(const VkSamplerCreateInfo &info) const;
  
  private:
    template <typename T>
    static inline void hash_combine(std::size_t &s, const T &v) {
      std::hash<T> h;
      s ^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2); 
    }
  };

  struct SamplerEqualFunc : std::binary_function<VkSamplerCreateInfo, VkSamplerCreateInfo, bool> {
    bool operator()(const VkSamplerCreateInfo &l, const VkSamplerCreateInfo &r) const;
  };

  struct SamplerPool {
    SamplerPool(VkDevice dev) : device {dev} {}
    ~SamplerPool();

    VkSampler get_sampler(const VkSamplerCreateInfo &info);

  private:
    SamplerPool(const SamplerPool &) = delete;
    SamplerPool &operator=(const SamplerPool &) = delete;

    VkDevice device = nullptr;
    std::unordered_map<VkSamplerCreateInfo, VkSampler, SamplerHashFunc, SamplerEqualFunc> samplers;
  };

  constexpr VkSamplerCreateInfo DEFAULT_SAMPLER {
    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    .pNext = nullptr,
    .flags = 0,
    .magFilter = VK_FILTER_LINEAR,
    .minFilter = VK_FILTER_LINEAR,
    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .mipLodBias = 0.f,
    .anisotropyEnable = VK_FALSE,
    .maxAnisotropy = 0.f,
    .compareEnable = VK_FALSE,
    .compareOp = VK_COMPARE_OP_ALWAYS,
    .minLod = 0.f,
    .maxLod = 10.f,
    .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    .unnormalizedCoordinates = VK_FALSE
  };

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

}

#endif
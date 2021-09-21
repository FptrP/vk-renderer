#include "samplers.hpp"

#include <cstring>

namespace gpu {

  std::size_t SamplerHashFunc::operator()(const VkSamplerCreateInfo &info) const {
    std::size_t h = 0;
    hash_combine(h, info.magFilter);
    hash_combine(h, info.minFilter);
    hash_combine(h, info.mipmapMode);
    hash_combine(h, info.addressModeU);
    hash_combine(h, info.addressModeV);
    hash_combine(h, info.addressModeW);
    hash_combine(h, info.mipLodBias);
    hash_combine(h, info.anisotropyEnable);
    hash_combine(h, info.maxAnisotropy);
    hash_combine(h, info.compareEnable);
    hash_combine(h, info.compareOp);
    hash_combine(h, info.minLod);
    hash_combine(h, info.maxLod);
    hash_combine(h, info.borderColor);
    hash_combine(h, info.unnormalizedCoordinates);
    return h;
  }

  bool SamplerEqualFunc::operator()(const VkSamplerCreateInfo &l, const VkSamplerCreateInfo &r) const {
    return !std::memcmp(&l, &r, sizeof(l));
  }

  SamplerPool::~SamplerPool() {
    if (!device)
      return;

    for (auto& [k, v] : samplers) {
      vkDestroySampler(device, v, nullptr);
    }
  }

  const SamplerPool &SamplerPool::operator=(SamplerPool &&pool) {
    std::swap(device, pool.device);
    samplers = std::move(pool.samplers);
    return *this;
  }

  VkSampler SamplerPool::get_sampler(const VkSamplerCreateInfo &info) {
    auto it = samplers.find(info);
    if (it != samplers.end()) {
      return it->second;
    }

    VkSampler new_sampler {nullptr};
    VKCHECK(vkCreateSampler(device, &info, nullptr, &new_sampler));
    samplers[info] = new_sampler;
    return new_sampler;
  }

}
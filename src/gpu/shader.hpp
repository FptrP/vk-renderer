#ifndef SHADER_HPP_INCLUDED
#define SHADER_HPP_INCLUDED

#include <array>
#include <vector>
#include <unordered_map>
#include <map>
#include <optional>

#include "lib/spirv-cross/spirv_cross.hpp"
#include "driver.hpp"

namespace gpu {

  struct DescriptorPool {
    DescriptorPool(uint32_t flips_count);
    ~DescriptorPool();

    void flip();
    void allocate_sets(uint32_t sets_count, const VkDescriptorSetLayout *set_layouts, VkDescriptorSet *sets, void *ext = nullptr);
    
    VkDescriptorSet allocate_set(VkDescriptorSetLayout set_layout) {
      VkDescriptorSet out {VK_NULL_HANDLE};
      allocate_sets(1, &set_layout, &out);
      return out;
    }

    VkDescriptorSet allocate_set(VkDescriptorSetLayout set_layout, const std::vector<uint32_t> &variable_sizes) {
      VkDescriptorSetVariableDescriptorCountAllocateInfo ext {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorSetCount = (uint32_t)variable_sizes.size(),
        .pDescriptorCounts = variable_sizes.data()
      };

      VkDescriptorSet out {VK_NULL_HANDLE};
      allocate_sets(1, &set_layout, &out, &ext);
      return out;
    }

    DescriptorPool(DescriptorPool &&o) : pools {std::move(o.pools)}, index {o.index} {}
  
    const DescriptorPool &operator=(DescriptorPool &&o) {
      std::swap(pools, o.pools);
      std::swap(index, o.index);
      return *this;
    }

    VkDescriptorPool current_pool() const { return pools[index]; }
    
  private:
    std::vector<VkDescriptorPool> pools;
    uint32_t index = 0;
  
    DescriptorPool(DescriptorPool&)=delete;
    const DescriptorPool &operator=(const DescriptorPool&)=delete;
  };  
}


#endif
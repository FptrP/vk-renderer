#include "shader.hpp"

#include <iostream>
#include <fstream>
#include <cassert>

#include "lib/spirv-cross/spirv_glsl.hpp"
#include "lib/spirv-cross/spirv_cross.hpp"

#include <sstream>
#include <initializer_list>

namespace gpu {
  

  DescriptorPool::DescriptorPool(uint32_t flips_count) 
  {
    VkDescriptorPoolSize sizes[] {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 512},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
      {VK_DESCRIPTOR_TYPE_SAMPLER, 512}
    };

    VkDescriptorPoolCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = 512,
      .poolSizeCount = sizeof(sizes)/sizeof(sizes[0]),
      .pPoolSizes = sizes
    };
    
    pools.resize(flips_count);
    
    for (uint32_t i = 0; i < flips_count; i++) {
      VKCHECK(vkCreateDescriptorPool(internal::app_vk_device(), &info, nullptr, &pools[i]));
    }
  }
  
  DescriptorPool::~DescriptorPool() {
    for (uint32_t i = 0; i < pools.size(); i++) {
      vkDestroyDescriptorPool(internal::app_vk_device(), pools[i], nullptr);
    }
  }

  void DescriptorPool::flip() {
    index = (index + 1) % pools.size();
    VKCHECK(vkResetDescriptorPool(internal::app_vk_device(), pools[index], 0));
  }
  
  void DescriptorPool::allocate_sets(uint32_t sets_count, const VkDescriptorSetLayout *set_layouts, VkDescriptorSet *sets, void *ext) {
    
    VkDescriptorSetAllocateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = ext,
      .descriptorPool = pools[index],
      .descriptorSetCount = sets_count,
      .pSetLayouts = set_layouts
    };

    VKCHECK(vkAllocateDescriptorSets(internal::app_vk_device(), &info, sets));
  }
}
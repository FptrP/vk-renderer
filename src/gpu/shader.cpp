#include "shader.hpp"

#include <iostream>
#include <fstream>
#include <cassert>

#include "lib/spirv-cross/spirv_glsl.hpp"
#include "lib/spirv-cross/spirv_cross.hpp"

#include <sstream>
#include <initializer_list>

namespace gpu {
  

  DescriptorPool::DescriptorPool(VkDevice base, uint32_t flips_count) 
    : device {base}
  {
    VkDescriptorPoolSize sizes[] {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 128},
      {VK_DESCRIPTOR_TYPE_SAMPLER, 128}
    };

    VkDescriptorPoolCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .maxSets = 256,
      .poolSizeCount = sizeof(sizes)/sizeof(sizes[0]),
      .pPoolSizes = sizes
    };
    
    pools.resize(flips_count);
    
    for (uint32_t i = 0; i < flips_count; i++) {
      VKCHECK(vkCreateDescriptorPool(device, &info, nullptr, &pools[i]));
    }
  }
  
  DescriptorPool::~DescriptorPool() {
    for (uint32_t i = 0; i < pools.size(); i++) {
      vkDestroyDescriptorPool(device, pools[i], nullptr);
    }
  }

  void DescriptorPool::flip() {
    index = (index + 1) % pools.size();
    VKCHECK(vkResetDescriptorPool(device, pools[index], 0));
  }
  
  void DescriptorPool::allocate_sets(uint32_t sets_count, const VkDescriptorSetLayout *set_layouts, VkDescriptorSet *sets) {
    
    VkDescriptorSetAllocateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = nullptr,
      .descriptorPool = pools[index],
      .descriptorSetCount = sets_count,
      .pSetLayouts = set_layouts
    };

    VKCHECK(vkAllocateDescriptorSets(device, &info, sets));
  }

  RenderSubpass::RenderSubpass(VkDevice device, std::initializer_list<VkFormat> color_attachments, std::optional<VkFormat> depth_attachment)
    : base {device}
  {
    std::vector<VkAttachmentDescription> attachments;
    
    VkAttachmentDescription desc {
      0,
      VK_FORMAT_R8G8B8A8_SRGB,
      VK_SAMPLE_COUNT_1_BIT,
      VK_ATTACHMENT_LOAD_OP_LOAD,
      VK_ATTACHMENT_STORE_OP_STORE,
      VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      VK_ATTACHMENT_STORE_OP_DONT_CARE,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    
    for (auto fmt : color_attachments) {
      desc.format = fmt;
      attachments.push_back(desc);
    }

    if (depth_attachment.has_value()) {
      desc.format = depth_attachment.value();
      desc.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      desc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachments.push_back(desc);
    }

    std::vector<VkAttachmentReference> references;
    for (uint32_t i = 0; i < color_attachments.size(); i++) {
      references.push_back({i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }

    VkAttachmentReference depth_reference {};
    if (depth_attachment.has_value()) {
      depth_reference = {(uint32_t)attachments.size() - 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    }

    VkSubpassDescription subpass {};
    subpass.colorAttachmentCount = references.size();
    subpass.pColorAttachments = references.data();
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (depth_attachment.has_value()) {
      subpass.pDepthStencilAttachment = &depth_reference;
    }

    VkRenderPassCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = attachments.size();
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    
    VKCHECK(vkCreateRenderPass(base, &info, nullptr, &handle));
  }

  
  static std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
  }

  static VkShaderModule create_shader_module(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VKCHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
  }

  ShaderModule::ShaderModule(VkDevice device, const char *path, VkShaderStageFlagBits shader_stage)
    : base {device}, stage {shader_stage}
  {
    auto bytes = read_file(path);
    assert(bytes.size() % sizeof(uint32_t) == 0);

    mod = create_shader_module(base, bytes);

    spirv_cross::Compiler compiler {(const uint32_t*)bytes.data(), bytes.size()/sizeof(uint32_t)};
    auto resources = compiler.get_shader_resources();
    
    add_resources(compiler, resources.storage_images.data(), resources.storage_images.size(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    add_resources(compiler, resources.storage_buffers.data(), resources.storage_buffers.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add_resources(compiler, resources.uniform_buffers.data(), resources.uniform_buffers.size(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
    add_resources(compiler, resources.sampled_images.data(), resources.sampled_images.size(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    if (resources.push_constant_buffers.size() > 1) {
      throw std::runtime_error {"Unsupported push_countants layout"};
    }

    if (resources.push_constant_buffers.size()) {
      const auto &resource = resources.push_constant_buffers[0];
      const auto &type = compiler.get_type(resource.base_type_id);
      size_t size = compiler.get_declared_struct_size(type);
      VkPushConstantRange r {};
      r.size = size;
      r.stageFlags = shader_stage;
      push_const = r;
    }
  }

  ShaderModule::~ShaderModule() {
    if (mod) {
      vkDestroyShaderModule(base, mod, nullptr);
      mod = nullptr;
    }
  }

  void ShaderModule::add_resources(const spirv_cross::Compiler &comp, const spirv_cross::Resource *res, uint32_t count, VkDescriptorType desc_type) {
    for (uint32_t i = 0; i < count; i++) {
      auto resource = res[i];
      auto set = comp.get_decoration(resource.id, spv::DecorationDescriptorSet);
      auto binding = comp.get_decoration(resource.id, spv::DecorationBinding);  

      VkDescriptorSetLayoutBinding api_binding {
        .binding = binding,
        .descriptorType = desc_type,
        .descriptorCount = 1,
        .stageFlags = stage,
        .pImmutableSamplers = nullptr
      };

      bindings[set].push_back(api_binding);
    }
  }

  static std::unordered_map<uint32_t, VkDescriptorSetLayout> merge_layouts(VkDevice device, std::initializer_list<const ShaderModule*> modules) {

    std::unordered_map<uint32_t, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding>> layouts;

    for (const auto mod : modules) {
      for (const auto &set : mod->get_bindings()) {
        const uint32_t set_id = set.first;

        for (const auto &resource : set.second) {
          if (!layouts[set_id].count(resource.binding)) {
            layouts[set_id][resource.binding] = resource;
            continue;
          }

          auto &src = layouts[set_id][resource.binding];
          if (src.descriptorType != resource.descriptorType) {
            std::stringstream ss;
            ss << "Error, incompatible descriptor type in set " << set_id << " binding " << resource.binding;
            throw std::runtime_error {ss.str()};
          }

          src.stageFlags |= resource.stageFlags;
        }
      }
    }

    std::unordered_map<uint32_t, VkDescriptorSetLayout> result;

    for (const auto &set : layouts) {
      std::vector<VkDescriptorSetLayoutBinding> descriptors;
      for (const auto& binding : set.second) { descriptors.push_back(binding.second); } 
    
      VkDescriptorSetLayoutCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = (uint32_t)descriptors.size(),
        .pBindings = descriptors.data()
      };

      VkDescriptorSetLayout layout;
      VKCHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));
      result[set.first] = layout;
    }

    return result;
  }
  
  static std::optional<VkPushConstantRange> merge_constants(std::initializer_list<const ShaderModule*> modules) {
    std::optional<VkPushConstantRange> result {};

    for (auto mod : modules) {
      const auto &pc = mod->get_push_constants();
      if (!result.has_value()) {
        result = pc;
        continue;
      }

      if (!pc.has_value()) {
        continue;
      }

      if (result->offset != pc->offset || result->size != pc->size) {
        throw std::runtime_error {"Incompatible push constants descriptions"};
      }
      result->stageFlags |= pc->stageFlags;
    }
    return result;
  }

  Pipeline::~Pipeline() {
    close();
  }

  void Pipeline::init_compute(const ShaderModule &smod) {
    close();

    if (smod.get_shader_stage() != VK_SHADER_STAGE_COMPUTE_BIT) {
      throw std::runtime_error {"Error, attempt to create compute pipeline with non-compute shader"};
    }

    pipeline_type = VK_PIPELINE_BIND_POINT_COMPUTE;

    std::vector<VkDescriptorSetLayout> init_layouts;

    for (const auto &set : smod.get_bindings()) {
      VkDescriptorSetLayoutCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = (uint32_t)set.second.size(),
        .pBindings = set.second.data()
      };

      VkDescriptorSetLayout desc_layout {};
      VKCHECK(vkCreateDescriptorSetLayout(base, &info, nullptr, &desc_layout));
      descriptors[set.first] = desc_layout;
      init_layouts.push_back(desc_layout);
    }

    auto val = smod.get_push_constants().value_or(VkPushConstantRange{});

    VkPipelineLayoutCreateInfo layout_info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = (uint32_t)init_layouts.size(),
      .pSetLayouts = init_layouts.size()? init_layouts.data() : nullptr,
      .pushConstantRangeCount = smod.get_push_constants().has_value()? 1u : 0u,
      .pPushConstantRanges = smod.get_push_constants().has_value()? &val : nullptr 
    };

    VKCHECK(vkCreatePipelineLayout(base, &layout_info, nullptr, &layout));

    VkPipelineShaderStageCreateInfo stage {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = smod.get_api_module(),
      .pName = "main",
      .pSpecializationInfo = nullptr
    };

    VkComputePipelineCreateInfo pipeline_info {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = stage,
      .layout = layout,
      .basePipelineHandle = nullptr,
      .basePipelineIndex = 0
    };

    VKCHECK(vkCreateComputePipelines(base, nullptr, 1, &pipeline_info, nullptr, &handle));
  }
  
  void Pipeline::close() {
    if (is_empty()) return;

    vkDestroyPipelineLayout(base, layout, nullptr);

    for (auto elem : descriptors) {
      vkDestroyDescriptorSetLayout(base, elem.second, nullptr);
    }

    vkDestroyPipeline(base, handle, nullptr);

    descriptors.clear();
    layout = nullptr;
    handle = nullptr;
  }

  void Pipeline::bind(VkCommandBuffer cmd) const {
    vkCmdBindPipeline(cmd, pipeline_type, handle);
  }

  void Pipeline::init_gfx(const RenderSubpass &subpass, const ShaderModule &vertex, const ShaderModule &fragment, GraphicsPipelineDescriptor &state)
  {
    close();
    pipeline_type = VK_PIPELINE_BIND_POINT_GRAPHICS;

    descriptors = merge_layouts(base, {&vertex, &fragment});
    auto push_const = merge_constants({&vertex, &fragment});
    auto pc_range = push_const.value_or(VkPushConstantRange{});

    std::vector<VkDescriptorSetLayout> info_layouts;

    for (auto val : descriptors) {
      info_layouts.push_back(val.second);
    }

    VkPipelineLayoutCreateInfo layout_info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = (uint32_t)info_layouts.size(),
      .pSetLayouts = info_layouts.data(),
      .pushConstantRangeCount = push_const.has_value()? 1u : 0u,
      .pPushConstantRanges = push_const.has_value()? &pc_range : nullptr
    };

    VKCHECK(vkCreatePipelineLayout(base, &layout_info, nullptr, &layout));

    VkPipelineShaderStageCreateInfo stages[] {
      {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex.get_api_module(),
        .pName = "main",
        .pSpecializationInfo = nullptr
      },
      {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment.get_api_module(),
        .pName = "main",
        .pSpecializationInfo = nullptr
      }
    };

    VkGraphicsPipelineCreateInfo gfx_info {};
    state.set_params(gfx_info);
    gfx_info.renderPass = subpass.api_renderpass();
    gfx_info.subpass = 0;
    gfx_info.stageCount = 2;
    gfx_info.pStages = stages;
    gfx_info.layout = layout;

    VKCHECK(vkCreateGraphicsPipelines(base, nullptr, 1, &gfx_info, nullptr, &handle));
  }
}
#include "pipelines.hpp"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

#include <lib/spirv-reflect/spirv_reflect.h>

#include <sstream>
#include <initializer_list>
#include <map>

namespace gpu {

  PipelinePool::PipelinePool() {
    VkPipelineCacheCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .initialDataSize = 0,
      .pInitialData = nullptr
    };
    VKCHECK(vkCreatePipelineCache(internal::app_vk_device(), &info, nullptr, &vk_cache));
  }

  PipelinePool::~PipelinePool() {
    for (auto &[k, v] : compute_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), v.handle, nullptr);
    }

    for (auto &[k, v] : graphics_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), v.handle, nullptr);
    }

    if (vk_cache) {
      vkDestroyPipelineCache(internal::app_vk_device(), vk_cache, nullptr);
    }

    for (auto &subpass : allocated_subpasses) {
      if (!subpass.is_empty()) {
        vkDestroyRenderPass(internal::app_vk_device(), subpass.handle, nullptr);
      } 
    }
    
    for (auto &prog : allocated_programs) {
      destroy_program(prog);
    }
  }

  void PipelinePool::destroy_program(PipelinePool::ShaderProgram &prog) {
    prog.resources.reset(nullptr);    
    for (auto mod : prog.modules) {
      vkDestroyShaderModule(internal::app_vk_device(), mod, nullptr);
    }
  }

  struct ResourceBinding {
    VkDescriptorSetLayoutBinding binding;
    bool bindless;
  };

  #define SPVR_ASSER(res) if ((res) != SPV_REFLECT_RESULT_SUCCESS) throw std::runtime_error {"SPVReflect error"}

  ProgramResources::~ProgramResources() {
    if (prog_layout)
      vkDestroyPipelineLayout(internal::app_vk_device(), prog_layout, nullptr);
    
    for (auto layout : set_layouts)
      vkDestroyDescriptorSetLayout(internal::app_vk_device(), layout, nullptr);
  }

  VkShaderStageFlagBits ProgramResources::parse_shader(const uint32_t *code, uint32_t size) {
    SpvReflectShaderModule mod {};
    
    if (spvReflectCreateShaderModule(size, code, &mod) != SPV_REFLECT_RESULT_SUCCESS) {
      throw std::runtime_error {"Shader parsing error"};
    }

    VkShaderStageFlagBits stage = static_cast<VkShaderStageFlagBits>(mod.shader_stage);

    uint32_t count = 0;
    SPVR_ASSER(spvReflectEnumerateDescriptorSets(&mod, &count, nullptr));

    std::vector<SpvReflectDescriptorSet*> sets(count);
    SPVR_ASSER(spvReflectEnumerateDescriptorSets(&mod, &count, sets.data()));
    
    for (auto set : sets) {
      uint32_t index = 0;
      
      auto it = set_to_index.find(set->set);
      if (it == set_to_index.end()) {
        index = set_resources.size();
        set_to_index[set->set] = index;
        set_resources.emplace_back(set->set);    
      } else {
        index = it->second;
      }

      set_resources.at(index).parse_resources(stage, set);
    }

    if (mod.push_constant_block_count > 1) {
      throw std::runtime_error {"Only 1 push_const block is supported"};
    }

    if (mod.push_constant_block_count) {
      auto &blk = mod.push_constant_blocks[0];
      if (blk.offset != 0) {
        throw std::runtime_error {"PushConst offset n e 0"};
      }

      if (!push_consts.stageFlags) {
        push_consts.size = blk.size;
      } else if (push_consts.size != blk.size) {
        throw std::runtime_error {"PushConst size mismatch"};
      }

      push_consts.stageFlags |= stage;
    }

    spvReflectDestroyShaderModule(&mod);
    return stage;
  }

  const DescriptorSetResources &ProgramResources::get_resources(uint32_t set_id) const {
    auto it = set_to_index.find(set_id);
    
    if (it == set_to_index.end())
      throw std::runtime_error {"set_id out of bounds"};

    return set_resources.at(it->second);  
  }

  VkDescriptorSetLayout ProgramResources::get_desc_layout(uint32_t set_id) const {
     auto it = set_to_index.find(set_id);
    
    if (it == set_to_index.end())
      throw std::runtime_error {"set_id out of bounds"};

    return set_layouts.at(it->second);
  }

  std::optional<ResourceLocation> ProgramResources::find_resource(const std::string &name) const {
    auto it = names.find(name);
    if (it == names.end())
      return {};
    return it->second;
  }

  void ProgramResources::create_layout() {
    for (auto &res : set_resources) {
      auto layout = res.create_layout();
      set_layouts.push_back(layout);
    }

    VkPipelineLayoutCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = (uint32_t)set_layouts.size(),
      .pSetLayouts = set_layouts.data(),
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };

    if (push_consts.stageFlags) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = &push_consts;
    }

    VKCHECK(vkCreatePipelineLayout(internal::app_vk_device(), &info, nullptr, &prog_layout));
  }

  void ProgramResources::create_names_table() {
    for (const auto &set : set_resources) {
      for (uint32_t i = 0; i < set.input_names.size(); i++) {
        if (!set.input_names[i].length())
          continue;
        names[set.input_names[i]] = ResourceLocation {set.set_index, set.inputs[i].binding};
      }
    }
  }


  void DescriptorSetResources::parse_resources(VkShaderStageFlagBits stage, SpvReflectDescriptorSet *set) {
    for (uint32_t i = 0u; i < set->binding_count; i++) {
      const auto &spv_binding = *set->bindings[i];
      
      uint32_t spv_binding_count = 1;
      for (uint32_t i = 0; i < spv_binding.array.dims_count; i++) {
        spv_binding_count *= spv_binding.array.dims[i];
      }

      auto spv_desc_type = static_cast<VkDescriptorType>(spv_binding.descriptor_type);
      if (spv_desc_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        spv_desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;

      bool spv_bindless = spv_binding_count == 0;
      if (spv_bindless) {
        spv_binding_count = BINDLESS_DESC_COUNT;
      }

      auto it = bindings.find(spv_binding.binding);
      if (it != bindings.end()) {
        auto &api_binding = inputs[it->second];

        if (api_binding.descriptorType != spv_desc_type)  {
          throw std::runtime_error {"Incompatible desc. type"};
        }

        if (api_binding.descriptorCount != spv_binding_count) {
          throw std::runtime_error {"Bindings count mismatch"};
        }

        api_binding.stageFlags |= stage;
        continue;
      }

      VkDescriptorSetLayoutBinding api_binding {};
      api_binding.binding = spv_binding.binding;
      api_binding.descriptorType = spv_desc_type;
      api_binding.stageFlags = stage;
      api_binding.descriptorCount = spv_binding_count;
      
      auto index = inputs.size();
      
      inputs.push_back(api_binding);
      inputs_flags.push_back(spv_bindless? (VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT|VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT) : 0);
      input_names.push_back(spv_binding.name);
      
      bindings[spv_binding.binding] = index;
    }
  }

  const VkDescriptorSetLayoutBinding &DescriptorSetResources::get_binding(uint32_t binding) const {
    auto it = bindings.find(binding);
    if (it == bindings.end())
      throw std::runtime_error {"Binding not found"};
    return inputs.at(it->second);
  }

  VkDescriptorBindingFlags DescriptorSetResources::get_flags(uint32_t binding) const {
    auto it = bindings.find(binding);
    if (it == bindings.end())
      throw std::runtime_error {"Binding not found"};
    return inputs_flags.at(it->second);
  }

  const std::string &DescriptorSetResources::get_binding_name(uint32_t binding) const {
    auto it = bindings.find(binding);
    if (it == bindings.end())
      throw std::runtime_error {"Binding not found"};
    return input_names.at(it->second);
  }

  VkDescriptorSetLayout DescriptorSetResources::create_layout() {
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .pNext = nullptr,
      .bindingCount = (uint32_t)inputs_flags.size(),
      .pBindingFlags = inputs_flags.data()
    };

    VkDescriptorSetLayoutCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &flags_info,
      .flags = 0,
      .bindingCount = (uint32_t)inputs.size(),
      .pBindings = inputs.data()
    };

    VkDescriptorSetLayout layout = nullptr;
    VKCHECK(vkCreateDescriptorSetLayout(internal::app_vk_device(), &info, nullptr, &layout));
    return layout;
  }

  using LayoutBuilder = std::map<uint32_t, std::unordered_map<uint32_t, ResourceBinding>>;

  static void validate_bindings(std::vector<ShaderBinding> &bindings) {
    std::sort(bindings.begin(), bindings.end(), [](const auto &left, const auto &right){
      return left.stage < right.stage;
    });

    auto supported_shaders = 
      VK_SHADER_STAGE_VERTEX_BIT|
      VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT|
      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT|
      VK_SHADER_STAGE_GEOMETRY_BIT|
      VK_SHADER_STAGE_FRAGMENT_BIT|
      VK_SHADER_STAGE_COMPUTE_BIT;

    bool is_compute_pipeline = false;
    uint32_t usage_mask = 0;

    for (const auto &binding : bindings) {
      if ((binding.stage & supported_shaders) == 0) {
        throw std::runtime_error {"Error, unsupported shader"};
      }

      if (binding.stage & usage_mask) {
        throw std::runtime_error {"Error, multiple usage of same shader stage"};
      }

      is_compute_pipeline |= binding.stage == VK_SHADER_STAGE_COMPUTE_BIT;
      usage_mask |= binding.stage;
    }

    if (is_compute_pipeline && bindings.size() != 1) {
      throw std::runtime_error {"Error, usage of compute shader with other stages"};
    }
  }

  static std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      std::stringstream ss;
      ss << "Failed to open file " << filename;
      throw std::runtime_error {ss.str()};
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

  void PipelinePool::create_program(const std::string &name, std::vector<ShaderBinding> &&bindings) {
    if (programs.count(name)) {
      throw std::runtime_error {"Attemp to recreate shader program"};
    }
    
    validate_bindings(bindings);
    auto index = allocated_programs.size();

    programs[name] = index;
    allocated_programs.push_back(ShaderProgram {
      std::move(bindings),
      {},
      {},
    });
    create_program(allocated_programs[index]);
  }

  void PipelinePool::create_program(ShaderProgram &prog) {
    prog.resources.reset(new ProgramResources {});

    for (auto &shader : prog.modules) {
      vkDestroyShaderModule(internal::app_vk_device(), shader, nullptr);
    }

    std::vector<VkShaderModule> modules;
    VkShaderModule mod = nullptr;

    for (const auto &binding : prog.shader_info) {
      try {
        auto code = read_file(binding.path);
        mod = create_shader_module(internal::app_vk_device(), code);
        prog.resources->parse_shader((const uint32_t*)code.data(), code.size());
        modules.push_back(mod);
      }
      catch(...) {
        for (auto mod : modules) {
          vkDestroyShaderModule(internal::app_vk_device(), mod, nullptr);
        }
        if (mod) {
           vkDestroyShaderModule(internal::app_vk_device(), mod, nullptr);
        }
        throw;
      }
    }

    prog.resources->create_layout();
    prog.resources->create_names_table();
    prog.modules = std::move(modules);
  }

  void PipelinePool::reload_programs() {
    for (auto &desc : compute_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), desc.second.handle, nullptr);
      desc.second.handle = nullptr;
    }

    for (auto &desc : graphics_pipelines) {
      vkDestroyPipeline(internal::app_vk_device(), desc.second.handle, nullptr);
      desc.second.handle = nullptr;
    }
    
    shader_programs.reload();

    for (auto &shader : allocated_programs) {
      create_program(shader);
    }
  }

  uint32_t PipelinePool::get_subpass_index(const RenderSubpassDesc &desc) {
    if (!render_subpasses.count(desc)) {
      uint32_t index = allocated_subpasses.size();
      allocated_subpasses.push_back({desc, nullptr});
      render_subpasses[desc] = index;
      return index;
    }

    return render_subpasses[desc];
  }
  
  void PipelinePool::RenderSubpass::create_renderpass() {
    std::vector<VkAttachmentDescription> attachments;
    
    VkAttachmentDescription attach_desc {
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
    
    for (auto fmt : desc.formats) {
      attach_desc.format = fmt;
      attachments.push_back(attach_desc);
    }

    if (desc.use_depth) {
      attachments.back().initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      attachments.back().finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    std::vector<VkAttachmentReference> references;
    for (uint32_t i = 0; i < attachments.size(); i++) {
      references.push_back({i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }

    if (desc.use_depth) {
      references.back().layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass {};
    subpass.colorAttachmentCount = desc.use_depth? (references.size() - 1) : references.size();
    subpass.pColorAttachments = references.data();
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (desc.use_depth) {
      subpass.pDepthStencilAttachment = &references.back();
    }

    VkRenderPassCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = attachments.size();
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    
    VKCHECK(vkCreateRenderPass(internal::app_vk_device(), &info, nullptr, &handle));
  }

  VkRenderPass PipelinePool::get_subpass(uint32_t subpass_index) {
    auto &subpass = allocated_subpasses.at(subpass_index);
    if (subpass.is_empty()) {
      subpass.create_renderpass();
    }
    return subpass.handle;
  }

  const RenderSubpassDesc &PipelinePool::get_subpass_desc(uint32_t subpass_index) const {
    auto &subpass = allocated_subpasses.at(subpass_index);
    return subpass.desc;
  }

  uint32_t PipelinePool::get_program_index(const std::string &name) const {
    return programs.at(name);
  }
  PipelinePool::ShaderProgram &PipelinePool::get_program(uint32_t index) {
    return allocated_programs.at(index);
  }
  
  const PipelinePool::ShaderProgram &PipelinePool::get_program(uint32_t index) const {
    return allocated_programs.at(index);
  }

  uint32_t PipelinePool::get_vinput_index(const VertexInput &vinput) {
    auto it = vertex_input.find(vinput);
    if (it == vertex_input.end()) {
      uint32_t index = allocated_vinput.size();
      allocated_vinput.push_back(vinput);
      vertex_input.insert({vinput, index});
      return index;
    }

    return it->second;
  }
  
  const VertexInput &PipelinePool::get_vinput(uint32_t index) const {
    return allocated_vinput.at(index);
  }

  uint32_t PipelinePool::get_registers_index(const Registers &regs) {
    auto it = registers.find(regs);
    
    if (it == registers.end()) {
      uint32_t index = allocated_registers.size();
      allocated_registers.push_back(regs);
      registers.insert({regs, index});
      return index;
    }

    return it->second;
  }

  const Registers &PipelinePool::get_registers(uint32_t index) const {
    return allocated_registers.at(index);
  }

  void BasePipeline::set_program(const std::string &name) {
    program_id = pool->get_program_index(name);
  }

  VkDescriptorSetLayout BasePipeline::get_layout(uint32_t index) const {
    if (!program_id.has_value()) {
      throw std::runtime_error {"Pipeline not attached to program"};
    }
    const auto &prog = pool->get_program(program_id.value());
    return prog.resources->get_desc_layout(index);
  }
  
  VkPipelineLayout BasePipeline::get_pipeline_layout() const {
    if (!program_id.has_value()) {
      throw std::runtime_error {"Pipeline not attached to program"};
    }
    const auto &prog = pool->get_program(program_id.value());
    return prog.resources->get_pipeline_layout();
  }

  const ProgramResources &BasePipeline::get_resources() const {
    if (!program_id.has_value()) {
      throw std::runtime_error {"Pipeline not attached to program"};
    }
    const auto &prog = pool->get_program(program_id.value());
    return *prog.resources;
  }

  VkPipeline PipelinePool::get_pipeline(const ComputePipeline &pipeline) {
    auto &res = compute_pipelines[pipeline];
    if (res.handle) {
      return res.handle;
    }
    
    const auto &prog = get_program(pipeline.program_id.value());
    
    if (prog.modules.size() != 1 || prog.shader_info[0].stage != VK_SHADER_STAGE_COMPUTE_BIT) {
      throw std::runtime_error {"Not compute program"};
    }

    VkPipelineShaderStageCreateInfo stage {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = prog.modules[0],
      .pName = prog.shader_info[0].main.c_str(),
      .pSpecializationInfo = nullptr
    };

    VkComputePipelineCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stage = stage,
      .layout = prog.resources->get_pipeline_layout(),
      .basePipelineHandle = nullptr,
      .basePipelineIndex = 0
    };

    VKCHECK(vkCreateComputePipelines(internal::app_vk_device(), vk_cache, 1, &info, nullptr, &res.handle));
    return res.handle;
  }

  VkPipeline PipelinePool::get_pipeline(const GraphicsPipeline &pipeline) {
    auto &res = graphics_pipelines[pipeline];
    if (res.handle) {
      return res.handle;
    }

    const auto &prog = get_program(pipeline.program_id.value());
    const auto &regs = get_registers(pipeline.regs_index.value());
    const auto &vinput = get_vinput(pipeline.vertex_input.value());
    auto renderpass = get_renderpass(pipeline);
    const auto &rp_desc = get_subpass_desc(pipeline.render_subpass.value());

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    
    for (uint32_t i = 0; i < prog.modules.size(); i++) {
      
      VkPipelineShaderStageCreateInfo stage {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = prog.shader_info[i].stage,
        .module = prog.modules[i],
        .pName = prog.shader_info[i].main.c_str(),
        .pSpecializationInfo = nullptr
      };

      stages.push_back(stage);
    }

    VkPipelineColorBlendAttachmentState blend_attachment {
      .blendEnable = VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT
    };

    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;
    uint32_t att_count = rp_desc.use_depth? (rp_desc.formats.size() - 1) : rp_desc.formats.size();
    blend_attachments.insert(blend_attachments.begin(), att_count, blend_attachment);

    VkPipelineColorBlendStateCreateInfo blend_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_CLEAR,
      .attachmentCount = (uint32_t)blend_attachments.size(),
      .pAttachments = blend_attachments.data(),
      .blendConstants {1.f, 1.f, 1.f, 1.f}
    };
    
    VkDynamicState dyn_states[] {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .dynamicStateCount = 2,
      .pDynamicStates = dyn_states
    };
    
    VkPipelineVertexInputStateCreateInfo vertex_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .vertexBindingDescriptionCount = (uint32_t)vinput.bindings.size(),
      .pVertexBindingDescriptions = vinput.bindings.data(),
      .vertexAttributeDescriptionCount = (uint32_t)vinput.attributes.size(),
      .pVertexAttributeDescriptions = vinput.attributes.data()
    };

    VkViewport empty_vp {0.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    VkRect2D empty_scissor {{0, 0}, {1, 1}};

    VkPipelineViewportStateCreateInfo viewport_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = &empty_vp,
      .scissorCount = 1,
      .pScissors = &empty_scissor
    };

    VkPipelineMultisampleStateCreateInfo multisample {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = 0,
      .flags = 0,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 0.f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE
    };

    VkGraphicsPipelineCreateInfo info {};

    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.stageCount = (uint32_t)stages.size();
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_state;
    info.pInputAssemblyState = &regs.assembly;
    info.pRasterizationState = &regs.rasterization;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &regs.depth_stencil;
    info.pColorBlendState = &blend_state;
    info.pDynamicState = &dynamic_state;
    info.pViewportState = &viewport_state;
    info.pTessellationState = nullptr;
    info.renderPass = renderpass;
    info.subpass = 0;
    info.layout = prog.resources->get_pipeline_layout();

    VKCHECK(vkCreateGraphicsPipelines(internal::app_vk_device(), vk_cache, 1, &info, nullptr, &res.handle));

    return res.handle;
  }
  
  VkRenderPass PipelinePool::get_renderpass(const GraphicsPipeline &pipeline) {
    return get_subpass(pipeline.render_subpass.value());
  }

  VkPipeline ComputePipeline::get_pipeline() {
    return pool->get_pipeline(*this);
  }

  VkPipeline GraphicsPipeline::get_pipeline() {
    return pool->get_pipeline(*this);
  }
  
  VkRenderPass GraphicsPipeline::get_renderpass() {
    return pool->get_renderpass(*this);
  }

  const RenderSubpassDesc &GraphicsPipeline::get_renderpass_desc() const {
    return pool->get_subpass_desc(render_subpass.value());
  }

  void GraphicsPipeline::set_vertex_input(const VertexInput &vinput) {
    vertex_input = pool->get_vinput_index(vinput);
  }
  
  void GraphicsPipeline::set_registers(const Registers &regs) {
    regs_index = pool->get_registers_index(regs);
  }
  
  void GraphicsPipeline::set_rendersubpass(const RenderSubpassDesc &subpass) {
    render_subpass = pool->get_subpass_index(subpass);
  }


  bool operator==(const VkVertexInputBindingDescription &l, const VkVertexInputBindingDescription &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }

  bool operator==(const VkVertexInputAttributeDescription &l, const VkVertexInputAttributeDescription &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }
  
  bool operator==(const VkPipelineInputAssemblyStateCreateInfo &l, const VkPipelineInputAssemblyStateCreateInfo &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }

  bool operator==(const VkPipelineRasterizationStateCreateInfo &l, const VkPipelineRasterizationStateCreateInfo &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }
  
  bool operator==(const VkPipelineDepthStencilStateCreateInfo &l, const VkPipelineDepthStencilStateCreateInfo &r) {
    return !std::memcmp(&l, &r, sizeof(l));
  }

}
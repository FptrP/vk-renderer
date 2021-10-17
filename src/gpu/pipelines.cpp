#include "pipelines.hpp"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

#include "lib/spirv-cross/spirv_glsl.hpp"
#include "lib/spirv-cross/spirv_cross.hpp"

#include <sstream>
#include <initializer_list>

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
    vkDestroyPipelineLayout(internal::app_vk_device(), prog.pipeline_layout, nullptr);
    for (auto &pair : prog.dsl) {
      vkDestroyDescriptorSetLayout(internal::app_vk_device(), pair.second, nullptr);
    }
    for (auto mod : prog.modules) {
      vkDestroyShaderModule(internal::app_vk_device(), mod, nullptr);
    }
  }

  using LayoutBuilder = std::unordered_map<uint32_t, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding>>;

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

  static void add_resources(LayoutBuilder &builder, const spirv_cross::Compiler &comp, VkShaderStageFlagBits stage, const spirv_cross::Resource *res, uint32_t count, VkDescriptorType desc_type) {
    for (uint32_t i = 0; i < count; i++) {
      auto resource = res[i];
      auto set = comp.get_decoration(resource.id, spv::DecorationDescriptorSet);
      auto binding = comp.get_decoration(resource.id, spv::DecorationBinding);  

      const auto &type = comp.get_type(resource.type_id);
      uint32_t desc_count = 1;
      if (type.array.size()) {
        for (auto elem : type.array) {
          desc_count *= elem;
        }
      }

      if (!builder[set].count(binding)) { //first creation
        VkDescriptorSetLayoutBinding api_binding {
          .binding = binding,
          .descriptorType = desc_type,
          .descriptorCount = desc_count,
          .stageFlags = stage,
          .pImmutableSamplers = nullptr
        };
        builder[set][binding] = api_binding;
      }

      auto &desc = builder[set][binding];
      if (desc.descriptorType != desc_type) {
        throw std::runtime_error {"Incompatible type for descriptors"};
      }

      if (desc.descriptorCount != desc_count) {
        throw std::runtime_error {"Incompatible array declaration"};
      }

      desc.stageFlags |= stage;
    }
  }

  static VkShaderModule load_module(VkDevice api_device, const ShaderBinding &shader, LayoutBuilder &layouts, VkPushConstantRange &push_const) {
    auto code = read_file(shader.path);
    if (code.size() % sizeof(uint32_t)) {
      throw std::runtime_error {"Shader is not valid spirv-file"};
    }

    auto mod = create_shader_module(api_device, code);
    
    spirv_cross::Compiler compiler {(const uint32_t*)code.data(), code.size()/sizeof(uint32_t)};
    auto resources = compiler.get_shader_resources();

    try {
      add_resources(layouts, compiler, shader.stage, resources.storage_images.data(), resources.storage_images.size(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
      add_resources(layouts, compiler, shader.stage, resources.storage_buffers.data(), resources.storage_buffers.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      add_resources(layouts, compiler, shader.stage, resources.uniform_buffers.data(), resources.uniform_buffers.size(), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
      add_resources(layouts, compiler, shader.stage, resources.sampled_images.data(), resources.sampled_images.size(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
      add_resources(layouts, compiler, shader.stage, resources.separate_images.data(), resources.separate_images.size(), VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
      add_resources(layouts, compiler, shader.stage, resources.separate_samplers.data(), resources.separate_samplers.size(), VK_DESCRIPTOR_TYPE_SAMPLER);
    }
    catch(...) {
      vkDestroyShaderModule(api_device, mod, nullptr);
      throw;
    }

    if (resources.push_constant_buffers.size() > 1) {
      vkDestroyShaderModule(api_device, mod, nullptr);
      throw std::runtime_error {"Unsupported push_countants layout"};
    }

    if (resources.push_constant_buffers.size()) {
      const auto &resource = resources.push_constant_buffers[0];
      const auto &type = compiler.get_type(resource.base_type_id);
      size_t size = compiler.get_declared_struct_size(type);
      if (push_const.size && push_const.size != size) {
        vkDestroyShaderModule(api_device, mod, nullptr);
        throw std::runtime_error {"Conflicting declarations for push_constatnts"};
      }

      push_const.size = size;
      push_const.stageFlags |= shader.stage;
    }

    return mod;
  }

  void PipelinePool::create_program(const std::string &name, std::initializer_list<ShaderBinding> shaders) {
    if (programs.count(name)) {
      throw std::runtime_error {"Attemp to recreate shader program"};
    }
    
    std::vector<ShaderBinding> bindings {shaders.begin(), shaders.end()};
    validate_bindings(bindings);

    LayoutBuilder layout_builder;
    VkPushConstantRange push_const {0, 0, 0};
    std::vector<VkShaderModule> modules;
    
    for (const auto &binding : bindings) {
      try {
        modules.push_back(load_module(internal::app_vk_device(), binding, layout_builder, push_const));
      }
      catch(...) {
        for (auto mod : modules) {
          vkDestroyShaderModule(internal::app_vk_device(), mod, nullptr);
        }
        throw;
      }
    }

    std::unordered_map<uint32_t, VkDescriptorSetLayout> api_layouts;
    std::vector<VkDescriptorSetLayout> vec_layouts;
    for (const auto &[set_id, bindings] : layout_builder) {
      std::vector<VkDescriptorSetLayoutBinding> api_bindings;

      for (const auto &[id, binding] : bindings) {
        api_bindings.push_back(binding);
      }

      VkDescriptorSetLayoutCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = (uint32_t)api_bindings.size(),
        .pBindings = api_bindings.data()
      };

      VKCHECK(vkCreateDescriptorSetLayout(internal::app_vk_device(), &info, nullptr, &api_layouts[set_id]));
      vec_layouts.push_back(api_layouts[set_id]);
    }

    VkPipelineLayoutCreateInfo info {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .setLayoutCount = (uint32_t)api_layouts.size(),
      .pSetLayouts = vec_layouts.data(),
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = nullptr
    };

    if (push_const.stageFlags) {
      info.pushConstantRangeCount = 1;
      info.pPushConstantRanges = &push_const;
    }

    VkPipelineLayout pipeline_layout;
    VKCHECK(vkCreatePipelineLayout(internal::app_vk_device(), &info, nullptr, &pipeline_layout));
    
    auto index = allocated_programs.size();

    programs[name] = index;
    allocated_programs.push_back(ShaderProgram {
      std::move(bindings),
      std::move(modules),
      std::move(api_layouts),
      pipeline_layout
    });
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
    return prog.dsl.at(index);
  }
  
  VkPipelineLayout BasePipeline::get_pipeline_layout() const {
    if (!program_id.has_value()) {
      throw std::runtime_error {"Pipeline not attached to program"};
    }
    const auto &prog = pool->get_program(program_id.value());
    return prog.pipeline_layout;
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
      .layout = prog.pipeline_layout,
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
    info.layout = prog.pipeline_layout;

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
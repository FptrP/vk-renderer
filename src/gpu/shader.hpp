#ifndef SHADER_HPP_INCLUDED
#define SHADER_HPP_INCLUDED

#include <array>
#include <vector>
#include <unordered_map>
#include <map>
#include <optional>

#include "lib/spirv-cross/spirv_cross.hpp"
#include "vkerror.hpp"

namespace gpu {

  struct DescriptorPool {
    DescriptorPool(VkDevice base, uint32_t flips_count);
    ~DescriptorPool();

    void flip();
    void allocate_sets(uint32_t sets_count, const VkDescriptorSetLayout *set_layouts, VkDescriptorSet *sets);
    
    VkDescriptorSet allocate_set(VkDescriptorSetLayout set_layout) {
      VkDescriptorSet out {VK_NULL_HANDLE};
      allocate_sets(1, &set_layout, &out);
      return out;
    }

    DescriptorPool(DescriptorPool &&o) : device {o.device}, pools {std::move(o.pools)}, index {o.index} {}
  
    const DescriptorPool &operator=(DescriptorPool &&o) {
      std::swap(device, o.device);
      std::swap(pools, o.pools);
      std::swap(index, o.index);
      return *this;
    }

    VkDescriptorPool current_pool() const { return pools[index]; }
    
  private:
    VkDevice device {nullptr};
    std::vector<VkDescriptorPool> pools;
    uint32_t index = 0;
  
    DescriptorPool(DescriptorPool&)=delete;
    const DescriptorPool &operator=(const DescriptorPool&)=delete;
  };

  struct RenderSubpass {
    ~RenderSubpass() { if (base && handle) vkDestroyRenderPass(base, handle, nullptr); }
    
    RenderSubpass(VkDevice device, std::initializer_list<VkFormat> color_attachments)
      : RenderSubpass(device, color_attachments, std::optional<VkFormat> {}) {}

    RenderSubpass(VkDevice device, std::initializer_list<VkFormat> color_attachments, VkFormat depth_attachment)
      : RenderSubpass(device, color_attachments, std::optional<VkFormat> {depth_attachment}) {}

    RenderSubpass(RenderSubpass &&o) : base {o.base}, handle {o.handle} { o.handle = nullptr; }
    
    VkRenderPass api_renderpass() const { return handle; }

    const RenderSubpass &operator=(RenderSubpass &&s) {
      std::swap(base, s.base);
      std::swap(handle, s.handle);
      return *this;
    }

  private:
    VkDevice base {nullptr};
    VkRenderPass handle {};

    RenderSubpass(VkDevice device, std::initializer_list<VkFormat> color_attachments, std::optional<VkFormat> depth_attachment);
    RenderSubpass(const RenderSubpass&) = delete;
    const RenderSubpass &operator=(const RenderSubpass&) = delete;
  };

  struct Framebuffer {
    Framebuffer(VkDevice dev, const RenderSubpass &subpass, VkExtent3D range, std::initializer_list<VkImageView> attachments) : device {dev} {
      VkFramebufferCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = subpass.api_renderpass(),
        .attachmentCount = (uint32_t)attachments.size(),
        .pAttachments = attachments.begin(),
        .width = range.width,
        .height = range.height,
        .layers = range.depth
      };

      VKCHECK(vkCreateFramebuffer(device, &info, nullptr, &handle));
    }

    ~Framebuffer() {
      if (device && handle) {
        vkDestroyFramebuffer(device, handle, nullptr);
      }
    }

    VkFramebuffer api_framebuffer() const { return handle; }

    Framebuffer(Framebuffer &&f) : device {f.device}, handle {f.handle} { f.handle = nullptr; }
    const Framebuffer &operator=(Framebuffer &&f) {
      std::swap(device, f.device);
      std::swap(handle, f.handle);
      return *this;
    }

  private:
    VkDevice device {nullptr};
    VkFramebuffer handle {nullptr};

    Framebuffer(const Framebuffer&) = delete;
    const Framebuffer &operator=(const Framebuffer&) = delete;
  };

  struct ShaderModule {
    ShaderModule(VkDevice device, const char *path, VkShaderStageFlagBits shader_stage);
    ~ShaderModule();

    VkShaderModule get_api_module() const { return mod; }
    VkShaderStageFlagBits get_shader_stage() const { return stage; }

    const std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> &get_bindings() const { return bindings; }
    const std::optional<VkPushConstantRange> &get_push_constants() const { return push_const; } 

    ShaderModule(ShaderModule &&o)
      : base {o.base}, stage {o.stage}, mod {o.mod}, bindings {std::move(o.bindings)}, push_const {std::move(o.push_const)}
    {
      o.mod = nullptr;
    }

    const ShaderModule &operator=(ShaderModule &&o) {
      std::swap(base, o.base);
      std::swap(stage, o.stage);
      std::swap(mod, o.mod);
      std::swap(bindings, o.bindings);
      std::swap(push_const, o.push_const);
      return *this;
    }

  private:
    VkDevice base {nullptr};
    VkShaderStageFlagBits stage;
    VkShaderModule mod {nullptr};

    std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> bindings;
    std::optional<VkPushConstantRange> push_const {};

    void add_resources(const spirv_cross::Compiler &comp, const spirv_cross::Resource *res, uint32_t count, VkDescriptorType desc_type);
    
    ShaderModule(const ShaderModule&) = delete;
    const ShaderModule &operator=(const ShaderModule&) = delete;
  };


  struct GraphicsPipelineDescriptor {
    GraphicsPipelineDescriptor(uint32_t color_attachments) {
      
      VkPipelineColorBlendAttachmentState state {
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT
      };

      blend_state.attachmentCount = color_attachments;

      attachments.insert(attachments.begin(), color_attachments, state);
      dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);
      dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
    }
    
    GraphicsPipelineDescriptor &set_vertex_input(std::initializer_list<VkVertexInputBindingDescription> bindings, std::initializer_list<VkVertexInputAttributeDescription> attribs) {
      vertex_bindings.insert(vertex_bindings.begin(), bindings);
      vertex_attributes.insert(vertex_attributes.begin(), attribs);
      return *this;
    }

    void set_params(VkGraphicsPipelineCreateInfo &info) {
      blend_state.pAttachments = attachments.data();
      dynamic_state.dynamicStateCount = dynamic_states.size();
      dynamic_state.pDynamicStates = dynamic_states.data();

      viewport_state.pViewports = &empty_vp;
      viewport_state.pScissors = &empty_scissor;

      vertex_state.vertexBindingDescriptionCount = vertex_bindings.size();
      vertex_state.pVertexBindingDescriptions = vertex_bindings.data();
      vertex_state.vertexAttributeDescriptionCount = vertex_attributes.size();
      vertex_state.pVertexAttributeDescriptions = vertex_attributes.data();

      info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      info.pNext = nullptr;
      info.flags = 0;
      info.pVertexInputState = &vertex_state;
      info.pInputAssemblyState = &assembly;
      info.pRasterizationState = &rasterization;
      info.pMultisampleState = &multisample;
      info.pDepthStencilState = &depth_stencil;
      info.pColorBlendState = &blend_state;
      info.pDynamicState = &dynamic_state;
      info.pViewportState = &viewport_state;
      info.pTessellationState = nullptr;
    }

    VkPipelineInputAssemblyStateCreateInfo assembly {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE
    };

    VkPipelineRasterizationStateCreateInfo rasterization {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.f,
      .depthBiasClamp = 0.f,
      .depthBiasSlopeFactor = 0.f,
      .lineWidth = 1.f 
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

    VkPipelineDepthStencilStateCreateInfo depth_stencil {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
      .front {},
      .back {},
      .minDepthBounds = 0.f,
      .maxDepthBounds = 1.f
    };

    VkPipelineColorBlendStateCreateInfo blend_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_CLEAR,
      .attachmentCount = 0,
      .pAttachments = nullptr,
      .blendConstants {1.f, 1.f, 1.f, 1.f}
    };
    
    VkPipelineDynamicStateCreateInfo dynamic_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .dynamicStateCount = 0,
      .pDynamicStates = nullptr
    };
    
    VkPipelineVertexInputStateCreateInfo vertex_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .vertexBindingDescriptionCount = 0,
      .pVertexBindingDescriptions = nullptr,
      .vertexAttributeDescriptionCount = 0,
      .pVertexAttributeDescriptions = nullptr
    };

    VkPipelineViewportStateCreateInfo viewport_state {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr
    };

    VkViewport empty_vp {0.f, 0.f, 1.f, 1.f, 0.f, 1.f};
    VkRect2D empty_scissor {{0, 0}, {1, 1}};

    std::vector<VkPipelineColorBlendAttachmentState> attachments {};
    std::vector<VkDynamicState> dynamic_states;
    std::vector<VkVertexInputBindingDescription> vertex_bindings;
    std::vector<VkVertexInputAttributeDescription> vertex_attributes;
  };

  using ShaderDescriptorLayouts = std::unordered_map<uint32_t, VkDescriptorSetLayout>;

  struct Pipeline {
    Pipeline(VkDevice device) : base {device} {}
    ~Pipeline();

    void init_compute(const ShaderModule &smod);
    void init_gfx(const RenderSubpass &subpass, const ShaderModule &vertex, const ShaderModule &fragment, GraphicsPipelineDescriptor &state);
    void close();

    void bind(VkCommandBuffer cmd) const;

    VkDescriptorSetLayout get_descriptor_set_layout(uint32_t set_id) const { return descriptors.at(set_id); }
    VkPipelineLayout get_pipeline_layout() const { return layout; }

    Pipeline(Pipeline &&o) 
      : base {o.base}, pipeline_type {o.pipeline_type}, handle {o.handle}, 
        layout {o.layout}, descriptors {std::move(o.descriptors)} 
      {
        o.base = nullptr;
        o.handle = nullptr;
        o.layout = nullptr;
      }

    const Pipeline &operator=(Pipeline &&o) {
      std::swap(base, o.base);
      std::swap(pipeline_type, o.pipeline_type);
      std::swap(handle, o.handle);
      std::swap(layout, o.layout);
      std::swap(descriptors, o.descriptors);
      return *this;
    }

    VkPipelineBindPoint get_type() const { return pipeline_type; }

  private:
    VkDevice base {nullptr};
    VkPipelineBindPoint pipeline_type;
    VkPipeline handle {};
    VkPipelineLayout layout {};
    
    ShaderDescriptorLayouts descriptors;

    bool is_empty() const { return handle == nullptr; }

    Pipeline(Pipeline&) = delete;
    const Pipeline &operator=(const Pipeline&) = delete;
  };

  void clear_color_attachments();
  
}


#endif
#ifndef BUFFER_POOL_HPP_INCLUDED
#define BUFFER_POOL_HPP_INCLUDED

#include <vector>

#include "vkerror.hpp"

#include <vector>
#include <algorithm>
#include <memory>

#include "gpu/pipelines.hpp"

namespace gpu {

  struct CmdContext;

  struct CmdBufferPool {
    CmdBufferPool(VkDevice logical_device, uint32_t queue_family)
      : device {logical_device}
    {
      VkCommandPoolCreateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family
      };

      VKCHECK(vkCreateCommandPool(logical_device, &info, nullptr, &pool));
    }
    
    ~CmdBufferPool() {
      if (device && pool) {
        vkDestroyCommandPool(device, pool, nullptr);
      }
    }

    CmdBufferPool(CmdBufferPool &&o) : device {o.device}, pool {o.pool} { o.pool = nullptr; }
    
    const CmdBufferPool &operator=(CmdBufferPool &&o) {
      std::swap(device, o.device);
      std::swap(pool, o.pool);
      return *this;
    }

    std::vector<VkCommandBuffer> allocate(uint32_t count) {
      VkCommandBufferAllocateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count
      };

      std::vector<VkCommandBuffer> result;
      result.resize(count);
      VKCHECK(vkAllocateCommandBuffers(device, &info, result.data()));
      return result;
    }

    std::vector<CmdContext> allocate_contexts(uint32_t frames_count);

    VkCommandBuffer allocate() {
      VkCommandBufferAllocateInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
      };

      VkCommandBuffer result;
      VKCHECK(vkAllocateCommandBuffers(device, &info, &result));
      return result;
    }

  private:
    VkDevice device;
    VkCommandPool pool {VK_NULL_HANDLE};

    CmdBufferPool(const CmdBufferPool&) = delete;
    const CmdBufferPool &operator=(const CmdBufferPool&) = delete; 
  };


  struct CtxResource {
    CtxResource() {}
    virtual void destroy(VkDevice device) = 0;
    virtual ~CtxResource() {} 
  };

  struct CmdContext {
    CmdContext(VkDevice device, VkCommandBuffer cmd_buf) : api_device {device}, cmd {cmd_buf} {}
    ~CmdContext() { clear_resources(); }
    
    void begin();
    void end();

    void set_framebuffer(uint32_t width, uint32_t height, const std::initializer_list<VkImageView> &views);
    void bind_pipeline(const GraphicsPipeline &pipeline);
    void bind_pipeline(const ComputePipeline &pipeline);
    void end_renderpass();

    void bind_descriptors_compute(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets, const std::initializer_list<uint32_t> offsets);
    void bind_descriptors_graphics(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets, const std::initializer_list<uint32_t> offsets);
    void bind_descriptors_compute(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets);
    void bind_descriptors_graphics(uint32_t first_set, const std::initializer_list<VkDescriptorSet> &sets);

    void bind_viewport(VkViewport viewport);
    void bind_scissors(VkRect2D scissors);
    void bind_viewport(float x, float y, float w, float h, float min_d, float max_d) { bind_viewport({x, y, w, h, min_d, max_d}); }
    void bind_scissors(int32_t x, int32_t y, uint32_t w, uint32_t h) { bind_scissors({{x, y}, {w, h}}); }
    
    void clear_color_attachments(float r, float g, float b, float a);
    void clear_depth_attachment(float val);

    void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);
    void dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

    void push_constants_graphics(VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void *constants);
    void push_constants_compute(uint32_t offset, uint32_t size, const void *constants);
    
    VkCommandBuffer get_command_buffer() const { return cmd; }
    void clear_resources();
    //void draw_indexed
  private:
    VkDevice api_device = nullptr;
    VkCommandBuffer cmd = nullptr;

    std::optional<GraphicsPipeline> gfx_pipeline {};
    std::optional<ComputePipeline> cmp_pipeline {};

    struct BindedState {
      VkRenderPass renderpass = nullptr;
      VkFramebuffer framebuffer = nullptr;
      VkPipeline gfx_pipeline = nullptr;
      VkPipelineLayout gfx_layout = nullptr;
      VkPipeline cmp_pipeline = nullptr;
      VkPipelineLayout cmp_layout = nullptr;
    } state {};

    struct FramebufferState {
      bool dirty = true;
      uint32_t width;
      uint32_t height;
      std::vector<VkImageView> attachments;
    } fb_state {};

    std::vector<CtxResource*> delayed_free {};
  
    void flush_framebuffer_state(VkRenderPass renderpass);
    //CmdContext(const CmdContext &) = delete;
    //CmdContext(CmdContext &&) = default;
  };

}

#endif
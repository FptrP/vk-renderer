#ifndef GPU_FRAMEBUFFERS_HPP_INCLUDED
#define GPU_FRAMEBUFFERS_HPP_INCLUDED

#include "driver.hpp"
#include "managed_resources.hpp"

namespace gpu {

  struct FramebufferState {
    uint32_t width;
    uint32_t height;
    uint32_t layers;

    VkRenderPass renderpass;
    std::vector<VkImageView> views;
    std::vector<DriverResourceID> image_ids;
  };

  struct FramebuffersCache {
    FramebuffersCache();
    ~FramebuffersCache();

    VkFramebuffer get_framebuffer(FramebufferState &state);
    void gc();

  private:

  };

}


#endif
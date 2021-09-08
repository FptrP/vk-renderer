#ifndef GBUFFER_HPP_INCLUDED
#define GBUFFER_HPP_INCLUDED

#include "gpu/driver.hpp"

namespace renderer {

  struct Gbuffer {
    Gbuffer(const gpu::Device &device, VkExtent2D resolution);
    ~Gbuffer();

    const gpu::Image &get_albedo() const { return albedo; }
    const gpu::Image &get_normal() const { return normal; }
    const gpu::Image &get_material() const { return material; }
    const gpu::Image &get_depth() const { return depth; }

    gpu::Image &get_albedo() { return albedo; }
    gpu::Image &get_normal() { return normal; }
    gpu::Image &get_material() { return material; }
    gpu::Image &get_depth() { return depth; }

    const gpu::RenderSubpass &get_subpass() const { return subpass; }
    const gpu::Framebuffer &get_framebuffer() const { return framebuffer; }
  private:
    gpu::Image albedo;
    gpu::Image normal;
    gpu::Image material;
    gpu::Image depth;
    gpu::RenderSubpass subpass;
    gpu::Framebuffer framebuffer;
  };

  void start_gbuffer_subpass(VkCommandBuffer cmd, const Gbuffer &gbuffer);
  void end_gbuffer_subpass(VkCommandBuffer cmd, const Gbuffer &gbuffer);
  
}


#endif
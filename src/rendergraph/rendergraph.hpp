#ifndef RENDERGRAPH_HPP_INCLUDED
#define RENDERGRAPH_HPP_INCLUDED

#include <cinttypes>
#include <unordered_map>
#include "resources.hpp"

namespace rendergraph {
  struct RenderGraph;

  struct RenderGraphBuilder {
    //RenderGraphBuilder(RenderGraph &rg) : graph {rg} {}

    template <typename ImageRes>
    std::size_t use_color_attachment(uint32_t mip = 0, uint32_t layer = 0) {
      auto hash = get_image_hash<ImageRes>();
      use_color_attachment(hash, mip, layer);
      return hash;
    }

    template <typename ImageRes>
    std::size_t use_depth_attachment(uint32_t mip = 0, uint32_t layer = 0) {
      auto hash = get_image_hash<ImageRes>();
      use_depth_attachment(hash, mip, layer);
      return hash;
    }

    template <typename ImageRes>
    std::size_t sample_image(VkShaderStageFlags stages, uint32_t base_mip = 0, uint32_t mip_count = 1, uint32_t base_layer = 0, uint32_t layer_count = 1) {
      auto hash = get_image_hash<ImageRes>();
      sample_image(hash, stages, base_mip, mip_count, base_layer, layer_count);
      return hash;
    }

  private:
    //RenderGraph &graph;
    ResourceInput input;
  
    void use_color_attachment(std::size_t image_id, uint32_t mip, uint32_t layer);
    void use_depth_attachment(std::size_t image_id, uint32_t mip, uint32_t layer);
    void sample_image(std::size_t image_id, VkShaderStageFlags stages, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count);

  };


  /*struct GpuContext;
  struct GpuCmdContext;
  
  struct RenderGraphBuilder;

  struct BaseTask {
    virtual void create(GpuContext &, RenderGraphBuilder &) = 0;
    virtual void write_commands(GpuContext &, GpuCmdContext &) = 0;
    virtual ~BaseTask() {}
  };

  struct GpuContext {

  };
  
  struct GpuCmdContext {

  };

  template <typename TaskData>
  using TaskRunCB = std::function<void (TaskData &, GpuContext &, GpuCmdContext &)>;

  template <typename TaskData>
  using TaskCreateCB = std::function<void (TaskData &, GpuContext &, RenderGraphBuilder &)>;

  template <typename TaskData>
  struct Task : BaseTask {
    TaskData data;
    TaskCreateCB<TaskData> create_cb;
    TaskRunCB<TaskData> callback;

    void write_commands(GpuContext &ctx, GpuCmdContext &cmd) override {
      if (callback) {
        callback(data, ctx, cmd);
      }
    }

    void create(GpuContext &ctx, RenderGraphBuilder &builder) override {
      if (create_cb) {
        create_cb(data, ctx, builder);
      }
    }

  };

  struct GraphResources {
    std::unordered_map<std::size_t, uint32_t> image_remap;
    std::unordered_map<std::size_t, uint32_t> buffer_remap;
    std::vector<Image> images;
    std::vector<Buffer> buffers;
  };

  

  struct RenderGraph {
    

    void build();
  private:
    GraphResources resources;

    std::vector<std::unique_ptr<BaseTask>> tasks;
    std::vector<Barrier> barriers;

    void create_barriers(RenderGraphBuilder &&builder);
    BufferBarrierState& get_buffer_barrier();


    friend struct RenderGraphBuilder;
  };


  struct RenderGraphBuilder {
    RenderGraphBuilder(RenderGraph &rg) : graph {rg} {}

    template <typename ImageRes>
    std::size_t use_color_attachment(uint32_t mip = 0, uint32_t layer = 0) {
      auto hash = get_image_hash<ImageRes>();
      use_color_attachment(hash, mip, layer);
      return hash;
    }

    template <typename ImageRes>
    std::size_t use_depth_attachment(uint32_t mip = 0, uint32_t layer = 0) {
      auto hash = get_image_hash<ImageRes>();
      use_depth_attachment(hash, mip, layer);
      return hash;
    }

    template <typename ImageRes>
    std::size_t sample_image(uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) {
      auto hash = get_image_hash<ImageRes>();
      sample_image(base_mip, mip_count, base_layer, layer_count);
      return hash;
    }

  private:
    RenderGraph &graph;
    std::map<std::size_t, BufferState> buffers;
    std::map<ImageSubresource, ImageSubresourceState> images;
  
    void use_color_attachment(std::size_t image_id, uint32_t mip, uint32_t layer);
    void use_depth_attachment(std::size_t image_id, uint32_t mip, uint32_t layer);
    void sample_image(std::size_t image_id, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count);

  };*/

}

#endif

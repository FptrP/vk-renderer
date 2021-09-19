#ifndef RENDERGRAPH_HPP_INCLUDED
#define RENDERGRAPH_HPP_INCLUDED

#include <cinttypes>
#include <unordered_map>

#include "resources.hpp"
#include "gpu_ctx.hpp"
#include "gpu/descriptors.hpp"


namespace rendergraph {
  struct RenderGraph;

  struct RenderGraphBuilder {
    RenderGraphBuilder(GraphResources &res, GpuState &state, ImageResourceId backbuf)
      : resources {res}, gpu {state}, backbuffer {backbuf} {}

    void prepare_backbuffer();
    ImageViewId use_backbuffer_attachment();
    
    ImageViewId use_color_attachment(ImageResourceId id, uint32_t mip, uint32_t layer);
    ImageViewId use_depth_attachment(ImageResourceId id, uint32_t mip, uint32_t layer);
    ImageViewId sample_image(ImageResourceId id, VkShaderStageFlags stages, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count);
    
    const gpu::ImageInfo &get_image_info(ImageResourceId id);
    const ResourceInput &get_input() const { return input; }

    gpu::Device &get_gpu() { return gpu.get_device(); }
    uint32_t get_frames_count() const { return gpu.get_frames_count(); }
    uint32_t get_backbuffers_count() const { return gpu.get_backbuffers_count();}

  private:
    GraphResources &resources;
    GpuState &gpu;
    ResourceInput input;
    ImageResourceId backbuffer;

    friend struct RenderGraph;
  };

  struct RenderResources {
    RenderResources(GraphResources &res, GpuState &state) : resources {res}, gpu {state} {}

    gpu::Buffer &get_buffer(BufferResourceId id);
    gpu::Image &get_image(ImageResourceId id);
    VkImageView get_view(const ImageViewId &ref);

    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) { return gpu.allocate_set(layout); }
    VkDescriptorSet allocate_set(const gpu::Pipeline &pipeline, uint32_t index) { return gpu.allocate_set(pipeline.get_descriptor_set_layout(index)); }
    gpu::Device &get_gpu() const { return gpu.get_device(); }

    uint32_t get_frames_count() const { return gpu.get_frames_count(); }
    uint32_t get_backbuffers_count() const { return gpu.get_backbuffers_count();}
    uint32_t get_frame_index() const { return gpu.get_frame_index(); }
    uint32_t get_backbuffer_index() const { return gpu.get_backbuf_index(); }

  private:
    GraphResources &resources;
    GpuState &gpu;
  };

  struct BaseTask {
    BaseTask(const std::string &task_name) : name {task_name} {}
    virtual void write_commands(RenderResources &, gpu::CmdContext &) = 0;
    virtual ~BaseTask() {}

    std::string name;
  };

  template <typename TaskData>
  using TaskRunCB = std::function<void (TaskData &, RenderResources &, gpu::CmdContext &)>;

  template <typename TaskData>
  using TaskCreateCB = std::function<void (TaskData &, RenderGraphBuilder &)>;

  template <typename TaskData>
  struct Task : BaseTask {
    Task(const std::string &name) : BaseTask {name} {}

    TaskData data;
    TaskRunCB<TaskData> callback;

    void write_commands(RenderResources &resources, gpu::CmdContext &cmd) override {
      callback(data, resources, cmd);
    }
  };

  struct RenderGraph {
    RenderGraph(gpu::Device &device, gpu::Swapchain &swapchain);

    template <typename TaskData>
    void add_task(const std::string &name, TaskCreateCB<TaskData> create_cb, TaskRunCB<TaskData> run_cb) {
      RenderGraphBuilder builder {resources, gpu, get_backbuffer()};
      std::unique_ptr<Task<TaskData>> ptr {new Task<TaskData> {name}};
      create_cb(ptr->data, builder);
      ptr->callback = run_cb;
      tasks.push_back(std::move(ptr));
      tracking_state.add_input(builder.input, resources);
    }

    void submit();

    uint32_t get_frames_count() const { return gpu.get_frames_count(); }

    ImageResourceId create_image(VkImageType type, const gpu::ImageInfo &info, VkImageTiling tiling, VkImageUsageFlags usage);
    ImageResourceId create_image(const ImageDescriptor &desc);

  private:
    GpuState gpu;
    GraphResources resources;
    TrackingState tracking_state;

    std::vector<std::unique_ptr<BaseTask>> tasks;
    std::vector<ImageResourceId> backbuffers;

    void write_barrier(const Barrier &barrier, VkCommandBuffer cmd);
    ImageResourceId get_backbuffer() const;
    friend struct RenderGraphBuilder;
  };

}

#endif

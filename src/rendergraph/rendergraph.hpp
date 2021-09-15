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
    RenderGraphBuilder(GraphResources &res, GpuState &state) : resources {res}, gpu {state} {}

    template <typename ImageRes>
    ImageRef use_color_attachment(uint32_t mip = 0, uint32_t layer = 0) {
      auto hash = get_image_hash<ImageRes>();
      use_color_attachment(hash, mip, layer);
      return {hash, {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
    }

    template <typename ImageRes>
    ImageRef use_depth_attachment(uint32_t mip = 0, uint32_t layer = 0) {
      auto hash = get_image_hash<ImageRes>();
      use_depth_attachment(hash, mip, layer);
      return {hash, {VK_IMAGE_VIEW_TYPE_2D, mip, 1, layer, 1}};
    }

    template <typename ImageRes>
    ImageRef sample_image(VkShaderStageFlags stages, uint32_t base_mip = 0, uint32_t mip_count = 1, uint32_t base_layer = 0, uint32_t layer_count = 1) {
      auto hash = get_image_hash<ImageRes>();
      sample_image(hash, stages, base_mip, mip_count, base_layer, layer_count);
      return {hash, {VK_IMAGE_VIEW_TYPE_2D, base_mip, mip_count, base_layer, layer_count}};
    }

    void prepare_backbuffer();
    ImageRef use_backbuffer_attachment();

    const ResourceInput &get_input() const { return input; }

    template <typename ImageRes>
    std::size_t create_image(const ImageDescriptor &desc) {
      auto hash = get_image_hash<ImageRes>();
      create_img(hash, desc);
      return hash; 
    }

    template <typename BufferRes>
    void create_buffer(const BufferDescriptor &desc) {
      auto hash = get_buffer_hash<BufferRes>();
      create_buf(hash, desc);
      return hash; 
    }

    gpu::Device &get_gpu() { return gpu.get_device(); }
    
    uint32_t get_frames_count() const { return gpu.get_frames_count(); }
    uint32_t get_backbuffers_count() const { return gpu.get_backbuffers_count();}

  private:
    GraphResources &resources;
    GpuState &gpu;
    ResourceInput input;
  
    void use_color_attachment(std::size_t image_id, uint32_t mip, uint32_t layer);
    void use_depth_attachment(std::size_t image_id, uint32_t mip, uint32_t layer);
    void sample_image(std::size_t image_id, VkShaderStageFlags stages, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count);

    void create_img(std::size_t hash, const ImageDescriptor &desc);
    void create_buf(std::size_t hash, const BufferDescriptor &desc);

    friend struct RenderGraph;
  };

  struct RenderResources {
    RenderResources(GraphResources &res, GpuState &state) : resources {res}, gpu {state} {}

    gpu::Buffer &get_buffer(std::size_t id);
    gpu::Image &get_image(std::size_t id);
    VkImageView get_view(const ImageRef &ref);

    VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) { return gpu.allocate_set(layout); }
    
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
    virtual void write_commands(RenderResources &, VkCommandBuffer) = 0;
    virtual ~BaseTask() {}

    std::string name;
  };

  template <typename TaskData>
  using TaskRunCB = std::function<void (TaskData &, RenderResources &, VkCommandBuffer)>;

  template <typename TaskData>
  using TaskCreateCB = std::function<void (TaskData &, RenderGraphBuilder &)>;

  template <typename TaskData>
  struct Task : BaseTask {
    Task(const std::string &name) : BaseTask {name} {}

    TaskData data;
    TaskRunCB<TaskData> callback;

    void write_commands(RenderResources &resources, VkCommandBuffer cmd) override {
      callback(data, resources, cmd);
    }
  };

  struct RenderGraph {
    RenderGraph(gpu::Device &device, gpu::Swapchain &swapchain);

    template <typename TaskData>
    void add_task(const std::string &name, TaskCreateCB<TaskData> create_cb, TaskRunCB<TaskData> run_cb) {
      RenderGraphBuilder builder {resources, gpu};
      std::unique_ptr<Task<TaskData>> ptr {new Task<TaskData> {name}};
      create_cb(ptr->data, builder);
      ptr->callback = run_cb;
      tasks.push_back(std::move(ptr));
      tracking_state.add_input(builder.input);
    }

    void submit();

  private:
    GpuState gpu;
    GraphResources resources;
    TrackingState tracking_state;

    std::vector<std::unique_ptr<BaseTask>> tasks;
    std::vector<Barrier> barriers;

    void remap_backbuffer(); 
    void write_barrier(const Barrier &barrier, VkCommandBuffer cmd);
    friend struct RenderGraphBuilder;
  };

}

#endif

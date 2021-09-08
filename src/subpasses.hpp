#ifndef SUBPASSES_HPP_INCLUDED
#define SUBPASSES_HPP_INCLUDED

#include "framegraph.hpp"

struct BaseSubpass {

  /*void sample_image(uint32_t image_id, VkShaderStageFlags stages) {

  }*/

  void write_color_attachment(uint32_t image_id) {
    task.used_images.push_back({
      image_id, 0, 0,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    });
  }
  //void write_depth_attachment(uint32_t image_id);

  //void set_cb(TaskCB &&cb) { task_cb = std::move(cb); }
  
  uint32_t flush(RenderGraph &render_graph) {
    auto id = render_graph.add_task(std::move(task));
    //render_graph.set_callback(id, std::move(task_cb));
    return id;
  }

private:
  Task task; 
  TaskCB task_cb;
};

struct PresentPrepareSubpass {
  PresentPrepareSubpass(uint32_t image_id) {
    task.used_images.push_back({
      image_id, 0, 0,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      0,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    });
  }

  uint32_t flush(RenderGraph &render_graph) {
    auto id = render_graph.add_task(std::move(task));
    return id;
  }

private:
  Task task;
};

struct GraphicsSubpass {



};

struct ComputeSubpass {

};


#endif
#ifndef SCENE_HPP_INCLUDED
#define SCENE_HPP_INCLUDED

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <lib/volk.h>

#include <assimp/Importer.hpp>      
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/pbrmaterial.h>

#include "gpu/driver.hpp"
#include "gpu/pipelines.hpp"
namespace scene {

  struct Vertex {
    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 uv;
  };

  gpu::VertexInput get_vertex_input();

  struct Mesh {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
  };

  struct Scene {
    Scene (gpu::Device &device) : vertex_buffer {device.new_buffer()}, index_buffer {device.new_buffer()} {}

    void load(const std::string &path, const std::string &folder);
    void gen_buffers(gpu::Device &device);

    const gpu::Buffer &get_vertex_buffer() const { return  vertex_buffer; }
    const gpu::Buffer &get_index_buffer() const { return  index_buffer; }
    const std::vector<Mesh> &get_meshes() const { return meshes; }
     
  private:
    void process_meshes(const aiScene *scene);
    void process_materials(const aiScene *scene);
    void process_objects(const aiNode *node, glm::mat4 transform);

    std::vector<Mesh> meshes;
    std::vector<Vertex> verts;
    std::vector<uint32_t> indexes;

    gpu::Buffer vertex_buffer;
    gpu::Buffer index_buffer;

    std::string model_path;
  };

  gpu::Image load_image_rgba8(gpu::Device &device, gpu::TransferCmdPool &transfer_pool, const char *path);

}

#endif
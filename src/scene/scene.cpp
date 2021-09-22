#include "scene.hpp"


#include <iostream>

namespace scene {

  gpu::VertexInput get_vertex_input() {
    gpu::VertexInput vinput;

    vinput.bindings = {{0, sizeof(scene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    vinput.attributes = {
      {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, pos)
      },
      {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, norm)
      },
      {
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(scene::Vertex, uv)
      }
    };
    
    return vinput;
  }

  void Scene::load(const std::string &path, const std::string &folder) {
    Assimp::Importer importer {};
    auto aiscene = importer.ReadFile(path, aiProcess_GenSmoothNormals|aiProcess_Triangulate| aiProcess_SortByPType | aiProcess_FlipUVs);
    model_path = folder;

    //process_materials(aiscene);
    process_meshes(aiscene);
    //process_objects(aiscene->mRootNode, glm::identity<glm::mat4>());
  }

  void Scene::process_meshes(const aiScene *scene) {
    const uint32_t meshes_count = scene->mNumMeshes;
    uint32_t verts_count = 0, index_count = 0;

    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
      const auto &mesh = scene->mMeshes[i];
      verts_count += mesh->mNumVertices;
      index_count += 3 * mesh->mNumFaces;
    }

    meshes.reserve(meshes_count);
    verts.reserve(verts_count);
    indexes.reserve(index_count);

    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
      const auto &scene_mesh = scene->mMeshes[i];
    
      Mesh mesh;  
  
      mesh.vertex_offset = verts.size();
      mesh.index_offset = indexes.size();
      mesh.index_count = scene_mesh->mNumFaces * 3;
      meshes.push_back(mesh);

      for (uint32_t j = 0; j < scene_mesh->mNumVertices; j++) {
        Vertex vertex;
        vertex.pos = {scene_mesh->mVertices[j].x, scene_mesh->mVertices[j].y, scene_mesh->mVertices[j].z};
        vertex.norm = {scene_mesh->mNormals[j].x, scene_mesh->mNormals[j].y, scene_mesh->mNormals[j].z};
        vertex.uv = {scene_mesh->mTextureCoords[0][j].x, scene_mesh->mTextureCoords[0][j].y};
        verts.push_back(vertex);
      }

      for (uint32_t j = 0; j < scene_mesh->mNumFaces; j++) {
        assert(scene_mesh->mFaces[j].mNumIndices == 3);
        auto i1 = scene_mesh->mFaces[j].mIndices[0];
        auto i2 = scene_mesh->mFaces[j].mIndices[1];
        auto i3 = scene_mesh->mFaces[j].mIndices[2];
      
        indexes.push_back(i1);
        indexes.push_back(i2);
        indexes.push_back(i3);
      }
    }

    std::cout << "Total " << meshes_count << " meshes, " << verts_count << " vertices " << index_count << " indexes\n";
  }

  static void copy_data(gpu::Device &device, VkCommandBuffer cmd, gpu::Buffer &dst, gpu::Buffer &transfer, uint32_t byte_count, uint8_t *data) {
    
    auto queue = device.api_queue();
    auto fence = device.new_fence();
    auto api_fence = static_cast<VkFence>(fence);

    VkSubmitInfo submit_info {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr
    };

    VkCommandBufferBeginInfo begin_info {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = 0,
      .pInheritanceInfo = nullptr
    };
    
    uint32_t offset = 0;
    uint32_t remaining_size = byte_count;

    while (remaining_size) {
      auto chunk = std::min(remaining_size, (uint32_t)transfer.get_size());
      std::memcpy(transfer.get_mapped_ptr(), data, chunk);
      transfer.flush();
      
      VkBufferCopy region {
        .srcOffset = 0,
        .dstOffset = offset,
        .size = chunk
      };

      data += chunk;
      remaining_size -= chunk;
      offset += chunk;

      vkResetFences(device.api_device(), 1, &api_fence);
      vkResetCommandBuffer(cmd, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
      vkBeginCommandBuffer(cmd, &begin_info);
      vkCmdCopyBuffer(cmd, transfer.get_api_buffer(), dst.get_api_buffer(), 1, &region);
      vkEndCommandBuffer(cmd);
      vkQueueSubmit(queue, 1, &submit_info, api_fence);
      vkWaitForFences(device.api_device(), 1, &api_fence, VK_TRUE, ~0ull);
    }

  }


  void Scene::gen_buffers(gpu::Device &device) {
    vertex_buffer.create(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(Vertex) * verts.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    index_buffer.create(VMA_MEMORY_USAGE_GPU_ONLY, sizeof(uint32_t) * indexes.size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    const uint32_t TRANSFER_SIZE = 10 * 1024;
    auto transfer_buffer = device.new_buffer();
    transfer_buffer.create(VMA_MEMORY_USAGE_CPU_TO_GPU, TRANSFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    auto pool = device.new_command_pool();
    auto cmd = pool.allocate();
    
    copy_data(device, cmd, vertex_buffer, transfer_buffer, vertex_buffer.get_size(), (uint8_t*)verts.data());
    copy_data(device, cmd, index_buffer, transfer_buffer, index_buffer.get_size(), (uint8_t*)indexes.data());

    verts.clear();
    indexes.clear();
  }


}
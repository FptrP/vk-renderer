#include "scene.hpp"


#include <iostream>
#include <unordered_map>

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

  gpu::VertexInput get_vertex_input_shadow() {
    gpu::VertexInput vinput;

    vinput.bindings = {{0, sizeof(scene::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    vinput.attributes = {
      {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(scene::Vertex, pos)
      }
    };
    
    return vinput;
  }

  static void load_verts_memory(const aiScene *scene, std::vector<Mesh> &meshes, std::vector<Vertex> &verts, std::vector<uint32_t> &indexes) {
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
      mesh.material_index = scene_mesh->mMaterialIndex;
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
  
  static void copy_data(gpu::TransferCmdPool &transfer_pool, gpu::Buffer &dst, gpu::Buffer &transfer, uint32_t byte_count, uint8_t *data) {
    
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

      auto cmd = transfer_pool.get_cmd_buffer();

      vkBeginCommandBuffer(cmd, &begin_info);
      vkCmdCopyBuffer(cmd, transfer.get_api_buffer(), dst.get_api_buffer(), 1, &region);
      vkEndCommandBuffer(cmd);
      transfer_pool.submit_and_wait();
    }

  }

  static void load_verts(gpu::TransferCmdPool &transfer_pool, const aiScene *scene, CompiledScene &out_scene, bool for_ray_traing) {
    std::vector<Vertex> cpu_verts;
    std::vector<uint32_t> cpu_indexes;
    load_verts_memory(scene, out_scene.meshes, cpu_verts, cpu_indexes);

    const uint64_t verts_size = sizeof(Vertex) * cpu_verts.size();
    const uint64_t index_size = sizeof(uint32_t) * cpu_indexes.size();

    VkBufferUsageFlags ray_tracing_flags = 0; 
    if (for_ray_traing) {
      ray_tracing_flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }

    out_scene.vertex_buffer.create(VMA_MEMORY_USAGE_GPU_ONLY, verts_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|ray_tracing_flags);
    out_scene.index_buffer.create(VMA_MEMORY_USAGE_GPU_ONLY, index_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|ray_tracing_flags);

    const uint32_t TRANSFER_SIZE = 10 * 1024;
    gpu::Buffer transfer_buffer;
    transfer_buffer.create(VMA_MEMORY_USAGE_CPU_TO_GPU, TRANSFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    
    copy_data(transfer_pool, out_scene.vertex_buffer, transfer_buffer, verts_size, (uint8_t*)cpu_verts.data());
    copy_data(transfer_pool, out_scene.index_buffer, transfer_buffer, index_size, (uint8_t*)cpu_indexes.data());

    cpu_verts.clear();
    cpu_indexes.clear();
  }

  struct MaterialDesc {
    std::string albedo_path;
    std::string mr_path;
  };

  static uint32_t load_scene_image(gpu::TransferCmdPool &transfer_pool, const std::string &path, std::unordered_map<std::string, uint32_t> &loaded_images, CompiledScene &out_scene) {
    if (path.empty()) {
      return UINT32_MAX;
    }

    auto iter = loaded_images.find(path);
    if (iter != loaded_images.end()) {
      return iter->second;
    }
    
    auto img = load_image_rgba8(transfer_pool, path.c_str());
    auto index = out_scene.images.size();
    out_scene.images.push_back(std::move(img));
    loaded_images[path] = index;
    
    return index;
  }

  static void load_materials(gpu::TransferCmdPool &transfer_pool, const aiScene *scene, const std::string &model_path, CompiledScene &out_scene) {
    out_scene.materials.reserve(scene->mNumMaterials);

    std::cout << "Materials count " << scene->mNumMaterials << "\n";
    
    std::unordered_map<std::string, uint32_t> loaded_images;  

    for (uint32_t i = 0; i < scene->mNumMaterials; i++) {
      const auto &scene_mt = scene->mMaterials[i];
      uint32_t count = scene_mt->GetTextureCount(aiTextureType_DIFFUSE);

      MaterialDesc mat {};
      int mode;
      scene_mt->Get(AI_MATKEY_TEXFLAGS(aiTextureType_DIFFUSE, 0), mode);

      if (count) {
        aiString path;
        scene_mt->GetTexture(aiTextureType_DIFFUSE, 0, &path);
        mat.albedo_path = path.C_Str();

        if (mode == aiTextureFlags_UseAlpha) {
          std::cout << mat.albedo_path << "\n";
        }

        if (mat.albedo_path.length()) {
          mat.albedo_path = model_path + mat.albedo_path;
        }

        path.Clear();

        scene_mt->GetTexture(aiTextureType_UNKNOWN, 0, &path);
        mat.mr_path = path.C_Str();
      
        if (mat.mr_path.length()) {
          mat.mr_path = model_path + mat.mr_path;
        }
      } 

      Material material {};
      material.clip_alpha = (mode == aiTextureFlags_UseAlpha);
      material.albedo_tex_index = load_scene_image(transfer_pool, mat.albedo_path, loaded_images, out_scene);
      material.metalic_roughness_index = load_scene_image(transfer_pool, mat.mr_path, loaded_images, out_scene);
      out_scene.materials.push_back(material);
    }

  }

  static void load_nodes(const aiNode *node, std::unique_ptr<Node> &out_node)
  {
    if (!node) return;

    out_node.reset(new Node {});

    out_node->transform[0] = glm::vec4{node->mTransformation.a1, node->mTransformation.b1, node->mTransformation.c1, node->mTransformation.d1};
    out_node->transform[1] = glm::vec4{node->mTransformation.a2, node->mTransformation.b2, node->mTransformation.c2, node->mTransformation.d2};
    out_node->transform[2] = glm::vec4{node->mTransformation.a3, node->mTransformation.b3, node->mTransformation.c3, node->mTransformation.d3};
    out_node->transform[3] = glm::vec4{node->mTransformation.a4, node->mTransformation.b4, node->mTransformation.c4, node->mTransformation.d4};

    for (uint32_t i = 0; i < node->mNumMeshes; i++) {
      out_node->meshes.push_back(node->mMeshes[i]);
    }

    out_node->children.resize(node->mNumChildren);

    for (uint32_t i = 0; i < node->mNumChildren; i++) {
      load_nodes(node->mChildren[i], out_node->children[i]);
    }
  }

  CompiledScene load_gltf_scene(gpu::TransferCmdPool &transfer_pool, const std::string &path, const std::string &folder, bool for_ray_traing) {
    CompiledScene result_scene {};

    Assimp::Importer importer {};
    auto aiscene = importer.ReadFile(path, aiProcess_GenSmoothNormals|aiProcess_Triangulate| aiProcess_SortByPType | aiProcess_FlipUVs);
    load_verts(transfer_pool, aiscene, result_scene, for_ray_traing);
    load_materials(transfer_pool, aiscene, folder, result_scene);
    load_nodes(aiscene->mRootNode, result_scene.root);
    return result_scene;
  }
}
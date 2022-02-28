#version 460 core
#include <gbuffer_encode.glsl>

layout (location = 0) in vec3 in_normal;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec4 pos_after;
layout (location = 3) in vec4 pos_before;

layout (location = 0) out vec4 out_albedo;
layout (location = 1) out vec4 out_normal;
layout (location = 2) out vec4 out_material;
layout (location = 3) out vec2 velocity_vector;

layout (set = 0, binding = 2) uniform texture2D material_textures[64];
layout (set = 0, binding = 3) uniform sampler main_sampler;

layout (push_constant) uniform push_data {
  uint transform_index;
  uint albedo_index;
  uint mr_index;
  uint flags;
};

void main() {
  out_albedo = texture(sampler2D(material_textures[albedo_index], main_sampler), in_uv);
  
  if (out_albedo.a == 0) {
    discard;
  }

  out_normal = vec4(encode_normal(in_normal), 0, 0);
  out_material = texture(sampler2D(material_textures[mr_index], main_sampler), in_uv);
  velocity_vector = 0.5 * (pos_before.xy/pos_before.w - pos_after.xy/pos_after.w);
}
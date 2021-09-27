#version 460 
#include <gbuffer_encode.glsl>

layout (location = 0) out vec4 out_color;

layout (location = 0) in vec2 screen_uv;

layout (set = 0, binding = 0) uniform sampler2D albedo_tex;
layout (set = 0, binding = 1) uniform sampler2D normal_tex;
layout (set = 0, binding = 2) uniform sampler2D material_tex;
layout (set = 0, binding = 3) uniform sampler2D depth_tex;

layout (set = 0, binding = 4) uniform Constants {
  mat4 inverse_camera;
  mat4 shadow_mvp;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (set = 0, binding = 5) uniform sampler2D shadow_map;
layout (set = 0, binding = 6) uniform sampler2D SSAO_tex;

const vec3 LIGHT_POS = vec3(0, 2, 0);

void main() {
  vec3 normal = sample_gbuffer_normal(normal_tex, screen_uv);
  vec4 albedo = texture(albedo_tex, screen_uv);
  float depth = texture(depth_tex, screen_uv).r;

  vec3 camera_view_vec = reconstruct_view_vec(screen_uv, depth, fovy, aspect, znear, zfar);
  vec4 world_pos = inverse_camera * vec4(camera_view_vec, 1);

  vec4 shadow_NDC = shadow_mvp * world_pos;
  shadow_NDC /= shadow_NDC.w;

  vec2 shadow_uv = 0.5 * (shadow_NDC.xy) + vec2(0.5, 0.5);
  //shadow_uv.y = 1 - shadow_uv.y;
  float world_depth = shadow_NDC.z;
  float shadow_depth = texture(shadow_map, shadow_uv.xy).r;

  vec3 L = normalize(LIGHT_POS - world_pos.xyz);
  float shade = max(dot(normal, L), 0.f); 
  if (shadow_NDC.x > -1 && shadow_NDC.x < 1 && shadow_NDC.y > -1 && shadow_NDC.y < 1 && world_depth > 0 && world_depth < 1) {
    if (world_depth > shadow_depth + 0.0001) {
      shade = 0.1;
    }
  }


  float ao = texture(SSAO_tex, screen_uv).r;
  ao *= ao;
  vec3 color = albedo.rgb * clamp(shade + 0, 0.f, 1.f) * clamp(2 * ao, 0.f, 1.f);

  out_color = vec4(color, 0);
}
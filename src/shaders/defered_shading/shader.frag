#version 460 
#include <gbuffer_encode.glsl>
#include <screen_trace.glsl>
#include <brdf.glsl>

layout (location = 0) out vec4 out_color;

layout (location = 0) in vec2 screen_uv;

layout (set = 0, binding = 0) uniform sampler2D albedo_tex;
layout (set = 0, binding = 1) uniform sampler2D normal_tex;
layout (set = 0, binding = 2) uniform sampler2D material_tex;
layout (set = 0, binding = 3) uniform sampler2D depth_tex;

layout (set = 0, binding = 4) uniform Constants {
  mat4 inverse_camera;
  mat4 camera_mat;
  mat4 shadow_mvp;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (set = 0, binding = 5) uniform sampler2D shadow_map;
layout (set = 0, binding = 6) uniform sampler2D TRACE_TEX;


const vec3 LIGHT_POS = vec3(-1.85867, 5.81832, -0.247114);
const vec3 LIGHT_RADIANCE = vec3(1, 1, 1);
#define USE_OCCLUSION 1

void main() {
  vec3 normal = sample_gbuffer_normal(normal_tex, screen_uv);
  vec3 albedo = texture(albedo_tex, screen_uv).xyz;
  vec4 material = texture(material_tex, screen_uv);
  float depth = textureLod(depth_tex, screen_uv, 0).r;
#if USE_OCCLUSION
  vec4 trace_res = texture(TRACE_TEX, screen_uv);
  float occlusion = trace_res.a;
  vec3 reflection = trace_res.rgb;
#else
  vec3 reflection = vec3(0);
  float occlusion = 1;
#endif
  vec3 camera_view_vec = reconstruct_view_vec(screen_uv, depth, fovy, aspect, znear, zfar);
  vec3 world_pos = (inverse_camera * vec4(camera_view_vec, 1)).xyz;
  vec3 camera_pos = (inverse_camera * vec4(0, 0, 0, 1)).xyz;

  const float metallic = material.b;
  const float roughness = material.g;

  const vec3 V = normalize(camera_pos - world_pos);
  const vec3 N = normal;
  
  vec3 F0 = F0_approximation(albedo, metallic);
  vec3 Lo = vec3(0);

  vec3 L = normalize(LIGHT_POS - world_pos);
  vec3 H = normalize(V + L);

  float light_distance = length(LIGHT_POS - world_pos);
  vec3 radiance = LIGHT_RADIANCE * min(100/(light_distance * light_distance), 100.0);

  float NDF = DistributionGGX(N, H, roughness);        
  float G = GeometrySmith(N, V, L, roughness);      
  vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);  

  vec3 kS = F;
  vec3 kD = (vec3(1.0) - kS) * (1 - metallic);

  float NdotL = max(dot(N, L), 0);
  vec3 specular = (NDF * G * F)/(4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);

  Lo += (kD * albedo/PI + specular) * radiance * NdotL;
  vec3 color = occlusion * (vec3(0.03) * albedo + Lo);
  color += albedo * reflection;
  out_color = vec4(color, 0);
}
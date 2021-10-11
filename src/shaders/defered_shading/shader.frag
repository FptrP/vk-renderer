#version 460 
#include <gbuffer_encode.glsl>
#include <screen_trace.glsl>

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
layout (set = 0, binding = 6) uniform sampler2D SSAO_tex;

const vec3 LIGHT_POS = vec3(0, 2, -1);

#define HORIZON_SHADOWS 0
#define HIERARCHICAL_SHADOWS 1 

void main() {
  vec3 normal = sample_gbuffer_normal(normal_tex, screen_uv);
  vec4 albedo = texture(albedo_tex, screen_uv);
  float depth = textureLod(depth_tex, screen_uv, 0).r;

  vec3 camera_view_vec = reconstruct_view_vec(screen_uv, depth, fovy, aspect, znear, zfar);
  vec4 world_pos = inverse_camera * vec4(camera_view_vec, 1);

  vec4 shadow_NDC = shadow_mvp * world_pos;
  shadow_NDC /= shadow_NDC.w;

  vec2 shadow_uv = 0.5 * (shadow_NDC.xy) + vec2(0.5, 0.5);
  //shadow_uv.y = 1 - shadow_uv.y;
  float world_depth = shadow_NDC.z;
  float shadow_depth = texture(shadow_map, shadow_uv.xy).r;

  vec3 color = vec3(0);

  vec3 L = normalize(LIGHT_POS - world_pos.xyz);
  float shade = max(dot(normal, L), 0.f); 
  if (shadow_NDC.x > -1 && shadow_NDC.x < 1 && shadow_NDC.y > -1 && shadow_NDC.y < 1 && world_depth > 0 && world_depth < 1) {
    if (world_depth > shadow_depth + 0.0001) {
      shade = 0.1;
    }
  }

#if HORIZON_SHADOWS
  {
    vec4 light_camera = camera_mat * vec4(LIGHT_POS, 1);
    vec3 camera_dir = 0.3 * normalize(light_camera.xyz - camera_view_vec);
    
    vec3 light_screen = project_view_vec(camera_view_vec + camera_dir, fovy, aspect, znear, zfar);
    vec3 pixel_screen = vec3(screen_uv, depth);
    if (light_screen.x < 0 || light_screen.x > 1 || light_screen.y < 0 || light_screen.y > 1)
      light_screen = clip_screen(pixel_screen, light_screen);
    
    vec3 dir = light_screen - pixel_screen;
    vec2 extended_dir = clip_screen(pixel_screen.xy, pixel_screen.xy + 2 * normalize(dir.xy));
    vec3 hor = simple_find_horizon(depth_tex, pixel_screen, extended_dir - pixel_screen.xy, 100, 0.4, znear, zfar);
    vec3 hor_dir = hor - pixel_screen;
    vec3 abs_dir = abs(dir);
    
    float hor_z_norm = hor_dir.z/max(abs(hor_dir.x), abs(hor_dir.y));
    float dir_z_norm = dir.z/max(abs_dir.x, abs_dir.y);

    if (dir_z_norm > hor_z_norm) {
      shade = 0.1;
    }
  }
#else
  {
    vec4 light_camera = camera_mat * vec4(LIGHT_POS, 1);
    vec3 camera_dir = 0.3 * normalize(light_camera.xyz - camera_view_vec);
    
    vec3 light_screen = project_view_vec(camera_view_vec + camera_dir, fovy, aspect, znear, zfar);
    vec3 pixel_screen = vec3(screen_uv, depth);
    if (light_screen.x < 0 || light_screen.x > 1 || light_screen.y < 0 || light_screen.y > 1)
      light_screen = clip_screen(pixel_screen, light_screen);
    
    vec3 dir = light_screen - pixel_screen;
    bool hit = false;
    vec3 hit_ray;
    
  #if HIERARCHICAL_SHADOWS
    hit_ray = hierarchical_raymarch_bounded(depth_tex, pixel_screen, light_screen, 0, 30, hit);
  #else
    hit = simple_raymarch(depth_tex, pixel_screen, light_screen, 0, hit_ray);
  #endif
    if (hit) {
      vec3 hit_dir = hit_ray - pixel_screen;
      float dist = dot(hit_dir, dir)/dot(dir, dir);
      
      float depth = textureLod(depth_tex, hit_ray.xy, 0).x;
      float linear_depth = linearize_depth2(depth, znear, zfar);
      float hit_linear_depth = linearize_depth2(hit_ray.z, znear, zfar);
      
      if (dist <= 1 &&  abs(hit_linear_depth - linear_depth) < 0.3) {
        shade = 0.1;
      }
    }
  }


#endif


  color += albedo.rgb * clamp(shade + 0, 0.f, 1.f);
  float ao = texture(SSAO_tex, screen_uv).r;
  ao *= ao;
  //color *= clamp(2 * ao, 0.f, 1.f);

  out_color = vec4(color, 0);
}
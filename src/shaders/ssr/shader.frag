#version 460
#include <gbuffer_encode.glsl>

layout (location = 0) in vec2 screen_uv;

layout (set = 0, binding = 0) uniform sampler2D normal_tex;
layout (set = 0, binding = 1) uniform sampler2D depth_tex;
layout (set = 0, binding = 2) uniform sampler2D frame_tex;

layout (set = 0, binding = 3) uniform SSRParams {
  mat4 camera_normal;
  float fovy;
  float aspect;
  float znear;
  float zfar;
};

layout (location = 0) out vec4 out_reflection;

bool simple_raymarch(in sampler2D depth_tex, vec3 start, vec3 end, const int lod, out vec3 out_ray);
bool hiz_trace(in sampler2D depth_tex, vec3 start, vec3 end, out vec3 out_ray);

void main() {
  float pixel_depth = texture(depth_tex, screen_uv).x;
  vec3 pixel_normal_world = sample_gbuffer_normal(normal_tex, screen_uv);
  vec3 pixel_normal = normalize((camera_normal * vec4(pixel_normal_world, 0)).xyz);
  
  vec3 view_vec = reconstruct_view_vec(screen_uv, pixel_depth, fovy, aspect, znear, zfar);

  vec3 R = reflect(view_vec, pixel_normal);

  vec3 start = project_view_vec(view_vec + 0.0001 * pixel_normal, fovy, aspect, znear, zfar);
  vec3 p = project_view_vec(view_vec + R, fovy, aspect, znear, zfar);
  vec3 delta = normalize(p - start);

  if (abs(delta.z) < 0.0000001) {
    out_reflection = vec4(0, 0, 0, 0);
    return;  
  }

  float t_bound = (1-start.z)/delta.z;
  float u_bound = max((1 - start.x)/delta.x, -start.x/delta.x);
  float v_bound = max((1 - start.y)/delta.y, -start.y/delta.y); 
  t_bound = min(t_bound, min(u_bound, v_bound));

  vec3 end = start + t_bound * delta;
  delta = end - start;

  const int LOD = 2;
  vec3 out_ray;
  //if (simple_raymarch(depth_tex, start, end, LOD, out_ray))
  if (hiz_trace(depth_tex, start, end, out_ray))
  {
    vec3 hit_normal_world = sample_gbuffer_normal(normal_tex, out_ray.xy);
    vec3 hit_normal = (camera_normal * vec4(hit_normal_world, 0)).xyz;

    if (dot(hit_normal, R) > 0) {
      out_reflection = vec4(0, 0, 0, 0);
      return;
    }

    out_reflection = texture(frame_tex, out_ray.xy);
    return;
  }

  out_reflection = vec4(1, 0, 0, 0);
}

bool simple_raymarch(in sampler2D depth_tex, vec3 start, vec3 end, const int lod, out vec3 out_ray) {
  vec3 delta = end - start;
  vec2 tex_size = textureSize(depth_tex, lod);

  ivec2 pixel_dist = ivec2(tex_size * abs(delta.xy));
  int steps = max(pixel_dist.x, pixel_dist.y);
  vec3 vec_step = delta/steps;
  
  for (int i = 1; i < steps - 1; i++) {
    vec3 p = start + vec_step * i;
    float depth = texelFetch(depth_tex, ivec2(floor(p.xy * tex_size)), lod).x;

    if (p.z - 0.00001 > depth) {
      
      if (p.z - depth > 0.01) {
        return false;  
      }

      out_ray = p;
      return true;
    }
  }

  return false;
}

vec3 intersect_cell(vec3 o, vec3 d, vec2 cell, vec2 cell_count, vec2 cross_step, vec2 cross_offset) {
	vec2 cell_size = 1.0/cell_count;
  vec2 planes = cell/cell_count + cell_size * cross_step;
  vec2 solutions = (planes - o.xy)/d.xy;
  vec3 intersection = o + d * min(solutions.x, solutions.y);
  intersection.xy += (solutions.x < solutions.y)? vec2(cross_offset.x, 0) : vec2(0, cross_offset.y);
  return intersection;
}

const int MAX_LOD = 5;
const uint MAX_ITERATIONS = 100; 

bool hiz_trace(in sampler2D depth_tex, vec3 start, vec3 end, out vec3 out_ray) {
  uint iterations = 0;
  int lod = 2;
  ivec2 mip0_size = ivec2(textureSize(depth_tex, 0));

  vec3 ray = start; 
  vec3 ray_dir = end - start;
  ray_dir /= ray_dir.z;

  vec3 o = ray;
  vec3 d = (end - start)/(end - start).z;
  if (d.z < 0) return false;

  vec2 cross_step = vec2(d.x >= 0? 1 : -1, d.y >= 0? 1 : -1);
  vec2 cross_offset = cross_step/(64 * 1920);
  cross_step = clamp(cross_step, 0, 1);

  vec2 start_size = mip0_size/(1 << 2);
  ray = intersect_cell(o, d, vec2(ray.xy * start_size), start_size, cross_step, cross_offset);

  while (lod >= 0 && iterations < MAX_ITERATIONS) {
    if (ray.x < 0 || ray.x > 1 || ray.y < 0 || ray.y > 1)
      return false;

    ivec2 tex_size = mip0_size/(1 << lod);
    ivec2 cell = ivec2(floor(ray.xy * tex_size));

    float z = texelFetch(depth_tex, ivec2(cell), lod).x;
    
    vec3 next_ray = (z > ray.z)? (o + d * z) : ray;
    ivec2 next_cell = ivec2(floor(next_ray.xy * tex_size));

    bool crossed_cell = (next_cell.x != cell.x) || (next_cell.y != cell.y);  

    ray = crossed_cell? intersect_cell(o, d, vec2(cell), vec2(tex_size), cross_step, cross_offset) : next_ray;
    lod = crossed_cell? min(MAX_LOD, lod + 1) : (lod - 1);
    iterations++;
  }

  out_ray = ray;
  if (lod < 0) {
    return true;
  }
  return true;
}
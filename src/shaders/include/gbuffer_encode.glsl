#ifndef GBUFFER_ENCODE_GLSL_INCLUDED
#define GBUFFER_ENCODE_GLSL_INCLUDED


float sign_nz(in float k) {
  return (k >= 0.0) ? 1.0 : -1.0;
}

vec2 sign_nz(in vec2 v) {
  return vec2(sign_nz(v.x), sign_nz(v.y));
}

vec3 sign_nz(in vec3 v) {
  return vec3(sign_nz(v.x), sign_nz(v.y), sign_nz(v.z));
}

vec2 encode_normal(in vec3 v) 
{
  float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
  vec2 result = v.xy * (1.0 / l1norm);
  
  if (v.z < 0.0) {
    result = (1.0 - abs(result.yx)) * sign_nz(result.xy);
  }

  return 0.5f * result + vec2(0.5f, 0.5f);
}

vec3 decode_normal(in vec2 uv)
{
  uv = 2.f * (uv - vec2(0.5f, 0.5f));
  vec3 v = vec3(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
  if (v.z < 0.0) {
    v.xy = (1.0 - abs(v.yx)) * sign_nz(v.xy);
  }

  return normalize(v);
}

float linearize_depth(float d,float zNear,float zFar)
{
  return zNear * zFar / (zFar + d * (zNear - zFar));
}

vec3 reconstruct_uv(vec2 uv, float z_near, float z_far, float plane_dist, float d)
{
  float z = linearize_depth(d, z_near, z_far);
  uv = 2 * uv - vec2(1, 1);
  return vec3(uv * z/plane_dist, z);
}

#endif
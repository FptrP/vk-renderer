#version 460 core

layout (set = 0, binding = 0) uniform sampler2D target_tex;

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;

float linearize_depth(float d,float zNear,float zFar)
{
  return zNear * zFar / (zFar + d * (zNear - zFar));
}

void main() {
  //float depth = linearize_depth(texture(target_tex, in_uv).r, 0.01, 10.0);
  //out_color = vec4(depth, depth, depth, depth);
  out_color = texture(target_tex, in_uv);
}
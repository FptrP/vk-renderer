#version 460 core

vec2 verts[] = vec2[](
  vec2(-1, 1),
  vec2(1, 1),
  vec2(0, -1)
);

layout (set = 0, binding = 0) uniform GbufConst {
  mat4 mvp;
};

void main() {
  gl_Position = mvp * vec4(verts[gl_VertexIndex], 0, 1);
}
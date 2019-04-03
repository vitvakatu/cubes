#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in ivec4 a_Normal;

layout(location = 0) out vec3 v_Normal;
layout(location = 1) out vec3 v_HalfDir;

const vec4 u_CameraPos = vec4(1.5, -5.0, 3.0, 1.0);
const vec4 u_LightPos = vec4(0.0, -10.0, 10.0, 1.0);

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_Normal = vec3(u_Transform * a_Normal);
    vec3 world_pos = vec3(u_Transform * a_Pos);
    vec3 light_dir = normalize(u_LightPos.xyz - world_pos);
    vec3 camera_dir = normalize(u_CameraPos.xyz - world_pos);
    v_HalfDir = normalize(light_dir + camera_dir);
    gl_Position = u_Transform * a_Pos;
}

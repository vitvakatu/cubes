#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in ivec4 a_Normal;
layout(location = 2) in vec4 a_OffsetScale;
layout(location = 3) in vec4 a_Rotation;
layout(location = 4) in vec4 a_Color;

layout(location = 0) out vec4 v_Color;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec3 v_HalfDir;

const vec4 u_CameraPos = vec4(-1.8, -8.0, 3.0, 1.0);
const vec4 u_LightPos = vec4(0.0, -10.0, 10.0, 1.0);
const vec4 u_LightColor = vec4(1.0, 1.0, 1.0, 1.0);

vec3 rotate_vector(vec4 quat, vec3 vec) {
    return vec + 2.0 * cross(cross(vec, quat.xyz) - quat.w * vec, quat.xyz);
}

layout(set = 0, binding = 0) uniform Globals {
    mat4 u_Projection;
};

void main() {
    v_Color = a_Color * u_LightColor;
    v_Normal = rotate_vector(a_Rotation, vec3(a_Normal.xyz));
    vec3 world_pos = rotate_vector(a_Rotation, a_Pos.xyz) * a_OffsetScale.w + a_OffsetScale.xyz;
    vec3 light_dir = normalize(u_LightPos.xyz - world_pos);
    vec3 camera_dir = normalize(u_CameraPos.xyz - world_pos);
    v_HalfDir = normalize(light_dir + camera_dir);
    gl_Position = u_Projection * vec4(world_pos, 1.0);
}

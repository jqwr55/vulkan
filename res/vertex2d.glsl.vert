#version 450
#extension GL_ARB_gpu_shader_int64 : require

layout(set = 0, binding = 0) uniform GlobalRenderInfo {

    mat4 projectionViewMatrix;
    mat4 inverseProjectionViewMatrix;
    vec4 viewDir;
    vec4 viewPos;
    vec4 viewRight;
    vec4 viewUp;
    float time;

} globalInfo;

layout(set = 1, binding = 0) uniform sampler2D texture_addresses[];

layout(location = 0) in vec2 pos;
layout(location = 1) in uint col;

layout(location = 0) out vec3 fragCol;

void main() {

    float r = float((col >> 0) & 0xFF);
    float g = float((col >> 8) & 0xFF);
    float b = float((col >> 16) & 0xFF);

    fragCol = vec3(r,g,b);
    gl_Position = vec4(pos.xy * 0.02, 0.0, 1.0);
}
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

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

layout(location = 2) in vec3 row0;
layout(location = 3) in vec3 row1;
layout(location = 4) in vec3 row2;
layout(location = 5) in vec3 row3;
layout(location = 6) in uint textureIndex;

layout(location = 0) out vec2 fragUv;
layout(location = 1) out flat  uint texIndex;


vec2 positions[3] = vec2[](
    
    vec2(-0.5, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {

    vec3 transformed = (mat3(row0, row1, row2) * pos) + row3;
    gl_Position = globalInfo.projectionViewMatrix * vec4(transformed, 1.0);
    
    fragUv = uv;
    texIndex = textureIndex;
}
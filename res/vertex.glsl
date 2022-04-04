#version 450
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer u32_addr {
   	uint values[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer i32_addr {
   	int values[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer u64_addr {
   	uint64_t values[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer i64_addr {
   	int64_t values[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer f32_addr {
   	float values[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer v4_addr {
   	vec4 values[];
};


layout(set = 0, binding = 0) uniform GlobalRenderInfo {

    mat4 projectionViewMatrix;
    mat4 inverseProjectionViewMatrix;
    vec4 viewDir;
    vec4 viewPos;
    vec4 viewRight;
    vec4 viewUp;
    float time;
    v4_addr mem;
    v4_addr host;
    v4_addr device;

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

void main() {

    vec3 transformed = (mat3(row0, row1, row2) * pos) + row3;
    gl_Position = globalInfo.projectionViewMatrix * vec4(transformed, 1.0);

    fragUv = uv;
    texIndex = textureIndex;
}
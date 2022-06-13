#include <common.h>

#define VULKAN_DEBUG 1
#include <vulkan.h>
#include <debug.h>
#include <open_type_loader.h>
#include <math3d.h>

#include <dirent.h> 
#include <time.h>
#include <typeinfo>
#include <thread>
#include <pthread.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_true_type.h"

#include "tiny_obj_loader.h"
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>


LoadedInfo LoadOBJ(byte* base, byte const* mem, const char* file) {

    auto rotMat = glm::eulerAngleXYZ(0, 0, 0);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "../res/room.obj");

    auto mat3 = ComputeRotarionXMat4(ToRadian(90.f));
    Mat4<f32> mat4;
    mat4.bases[0] = {mat3.bases[0].x, mat3.bases[0].y, mat3.bases[0].z, 0};
    mat4.bases[1] = {mat3.bases[1].x, mat3.bases[1].y, mat3.bases[1].z, 0};
    mat4.bases[2] = {mat3.bases[2].x, mat3.bases[2].y, mat3.bases[2].z, 0};
    mat4.bases[3] = {mat3.bases[3].x, mat3.bases[3].y, mat3.bases[3].z, 1};

    auto top = mem;
    LoadedInfo ret;
    ret.vertexOffset = mem - base;

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            auto v = (Vertex*)top;
            top += sizeof(Vertex);

            vec<f32, 4> pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2],
                1
            };

            pos = mat4 * pos;

            v->pos = {
                pos.x,
                pos.y,
                pos.z,
            };

            v->uv = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.f - attrib.texcoords[2 * index.texcoord_index + 1]
            };
        }
    }

    ret.vertexSize = top - mem;
    ret.indexOffset = ret.vertexOffset + ret.vertexSize;
    mem = top;

    u32 i = 0;
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            auto in = (u32*)top;
            top += sizeof(u32);
            *in = i++;
        }
    }

    ret.indexSize = top - mem;

    return ret;
}

struct EngineState {
    Camera camera;
    Mat4<f32> projection;
    f32 time;
    u32 delta;
    bool fullscreen;

    LinearAllocator alloc;
    LinearAllocator scratch;

    u32 drawCount;
    u32 instCount;
    DrawInfo* draws;
    InstanceInfo* instances;
    GlobalRenderParams* globalArgs;

    xcb_connection_t* connection;
    xcb_context xcb_ctx;
    xkb_keyboard keyboard;
    coroutine coro;

    VkRenderPass renderPass;
    VkSurfaceKHR surface;
    VkCoreContext core;
    VkGPUContext gpu;
    VkFbo fbo;
    VkExecutionResources exeRes;
    VkProgram2 program;
};
void Update(xcb_connection_t* connection, xkb_keyboard* keyboard, xcb_context* ctx, EngineState *state, u64 deltaTime) {

    state->time += 0.01;
    state->delta++;

    consume_xcb_events(connection, keyboard, 1, ctx);

    ComputeCameraVelocity(&state->camera, ctx->keys, 0.0001 * deltaTime);
    state->camera.position = state->camera.position + state->camera.vel;
    state->camera.vel = state->camera.vel * 0.95;

    bool ctrl = (ctx->keys >> KEY_BIT_LEFT_CTRL) & 1;
    if(!ctrl) {

        f32 cursorX = ctx->cursors_x;
        f32 cursorY = ctx->cursors_y;
        ctx->cursors_x = ctx->width >> 1;
        ctx->cursors_y = ctx->height >> 1;
        xcb_warp_pointer(connection, 0, ctx->window, 0,0,0,0, ctx->cursors_x, ctx->cursors_y);
        f32 halfX = ctx->width >> 1;
        f32 halfY = ctx->height >> 1;

        f32 horizontalAngle = (halfX - cursorX) / halfX;
        f32 verticalAngle = (halfY - cursorY) / halfY;
        RotateCamera(&state->camera, verticalAngle, -horizontalAngle);

        xcb_flush(connection);
    }
}

void global_io_flush_wrapper(void* user, LinearAllocator* io) {

    write(STDOUT_FILENO, io->base, io->top);
    io->top = 0;
}

void InitEngineState(EngineState* state, xcb_connection_t* connection, void* mem, u32 size) {


    state->connection = connection;
    auto id0 = xcb_generate_id(connection);
    state->keyboard = make_xcb_keys(connection);
    state->xcb_ctx = make_xcb_context(connection, id0, 640, 480, "vk_render_window");
    xcb_warp_pointer(connection, 0, state->xcb_ctx.window, 0,0,0,0, state->xcb_ctx.width/2, state->xcb_ctx.height/2);

    state->camera.position = {2,2,2};
    state->camera.direction = normalize(vec<f32,3>{4, 0, 0} - state->camera.position);
    state->camera.vel = {0,0,0};
    state->time = 0;
    state->delta = 0;
    state->fullscreen = false;
    state->projection = ComputePerspectiveMat4(ToRadian(90.0f), state->xcb_ctx.width / (f32)state->xcb_ctx.height, 0.01f, 100.0f);

    state->alloc = make_linear_allocator(mem, size - Megabyte(32));
    state->scratch = make_linear_allocator((byte*)mem + state->alloc.cap, Megabyte(32));

    VkRenderContextConfig config;
    config.logMask = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    config.windowHeight          = 480;
    config.windowWidth           = 640;
    config.ioLogBufferSize       = Kilobyte(2);
    config.scractchSize          = Kilobyte(64);
    config.vkHeapSize            = Megabyte(32);
    config.uploadHeapSize        = Megabyte(128);
    config.gpuHeapSize           = Megabyte(128);
    config.gpuhHeapMaxAllocCount = 256;

    MakeVkCoreContext(&state->core, config, {}, &state->alloc);

    VkXcbSurfaceCreateInfoKHR xcbSurfaceCreateInfo{};
    xcbSurfaceCreateInfo.connection = connection;
    xcbSurfaceCreateInfo.window = state->xcb_ctx.window;
    xcbSurfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    VK_CALL(state->core.vkScratch, vkCreateXcbSurfaceKHR, state->core.vkInstance, &xcbSurfaceCreateInfo, &state->core.vkAllocator, &state->surface);

    VkPhysicalDevice gpus[2];
    auto gpuCount = GetGPUs(&state->core, gpus);
    auto selectedGPU = PickPhysicalDevice(&state->core, state->surface, gpuCount, gpus, DEVICE_EXTENSIONS, SIZE_OF_ARRAY(DEVICE_EXTENSIONS));
    state->gpu = MakeVkGPUContext(&state->core, selectedGPU, state->surface, config, {}, &state->alloc);

    auto depthFormat = FindDepthFormat(&state->core, state->gpu.device);
    auto colorFormat = GetSurfaceFormat(&state->core, state->gpu.device, state->surface).format;
    state->renderPass  = CreateRenderPass(&state->core, state->gpu.logicalDevice, colorFormat, depthFormat);
    state->fbo = MakeVkFbo(&state->core, &state->gpu, state->renderPass, state->surface, 3, state->connection, &state->xcb_ctx, {}, &state->alloc);
    state->exeRes = MakeVkExecutionResources(&state->core, &state->gpu, {}, &state->alloc);

    VkDescriptorSetLayout layouts[2]{state->exeRes.layout0, state->exeRes.layout1};
    auto subpasses = (PipelineSubpass*)linear_allocate(&state->alloc, sizeof(PipelineSubpass) + sizeof(PipelineInfo) * 2);
    state->program = {
        .subpassCount = 1,
        .renderPass   = state->renderPass,
        .subpasses    = subpasses
    };

    ScopedAllocator scoped(&state->alloc);
    PipelineDescriptor texturedPipelineDescriptor{};
    texturedPipelineDescriptor.bindpoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    texturedPipelineDescriptor.vertByteCode = {
        (byte*)linear_allocator_top(&state->alloc),
        (u32)ReadFile("./vertex3d.spv", &state->alloc)
    };
    texturedPipelineDescriptor.fragByteCode = {
        (byte*)linear_allocator_top(&state->alloc),
        (u32)ReadFile("./textured.spv", &state->alloc)
    };
    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 0, 0, VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex, pos), &state->alloc);
    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 1, 0, VK_FORMAT_R32G32_SFLOAT,   offsetof(Vertex, uv), &state->alloc);

    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 2, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(v3) * 0, &state->alloc);
    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 3, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(v3) * 1, &state->alloc);
    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 4, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(v3) * 2, &state->alloc);
    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 5, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(v3) * 3, &state->alloc);
    PushAttribute(&texturedPipelineDescriptor.attribDescriptor, 6, 1, VK_FORMAT_R32_UINT, sizeof(v3) * 4, &state->alloc);

    PushAttributeBinding(&texturedPipelineDescriptor.attribDescriptor, 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX, &state->alloc);
    PushAttributeBinding(&texturedPipelineDescriptor.attribDescriptor, 1, sizeof(v3) * 4 + sizeof(u32), VK_VERTEX_INPUT_RATE_INSTANCE, &state->alloc);

    VkViewport viewPort = {0,0, 640,480 ,0,0};
    VkRect2D   scissor  = { {0,0}, {640,480} };
    VkDynamicState dynamic[2]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT |
                                VK_COLOR_COMPONENT_A_BIT;
    texturedPipelineDescriptor.inputAsm           = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, 0,0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false};
    texturedPipelineDescriptor.tessellationState  = {};
    texturedPipelineDescriptor.viewportState      = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,      0,0, 1, &viewPort, 1, &scissor};
    
    texturedPipelineDescriptor.rasterizationState.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    texturedPipelineDescriptor.rasterizationState.depthClampEnable        = VK_FALSE;
    texturedPipelineDescriptor.rasterizationState.rasterizerDiscardEnable = VK_FALSE;
    texturedPipelineDescriptor.rasterizationState.polygonMode             = VK_POLYGON_MODE_FILL;
    texturedPipelineDescriptor.rasterizationState.lineWidth               = 1.0f;
    texturedPipelineDescriptor.rasterizationState.depthBiasEnable         = VK_FALSE;
    texturedPipelineDescriptor.rasterizationState.cullMode                = VK_CULL_MODE_NONE;
    texturedPipelineDescriptor.rasterizationState.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    texturedPipelineDescriptor.multisampleState   = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,   0,0, VK_SAMPLE_COUNT_1_BIT, false, 0,0,0,0};
    texturedPipelineDescriptor.depthStencilState  = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, 0,0, true,true,VK_COMPARE_OP_LESS,false,false, {}, {}, 0,1.0};
    texturedPipelineDescriptor.colorBlendState    = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,   0,0, 0,VK_LOGIC_OP_COPY, 1, &colorBlend, {0.0, 0.0, 0.0, 0.0}};
    texturedPipelineDescriptor.dynamicState       = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,       0,0, 2, dynamic};

    state->program.subpasses[0].bindpoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    state->program.subpasses[0].pipelineCount = 2;
    state->program.subpasses[0].pipelines[0] = MakePipeline(&state->core, &state->gpu, 0, state->renderPass, 2, layouts, &global_debug_flat2d_pipeline);
    state->program.subpasses[0].pipelines[1] = MakePipeline(&state->core, &state->gpu, 0, state->renderPass, 2, layouts, &texturedPipelineDescriptor);
}
byte* InitState(byte* mem, u32 memSize, xcb_context xcb, EngineState* state) {

    i32 monitorCount = 0;
    state->camera.position = {2,2,2};
    state->camera.direction = normalize(vec<f32,3>{4, 0, 0} - state->camera.position);
    state->camera.vel = {0,0,0};

    state->time = 0;
    state->delta = 0;
    state->fullscreen = false;
    state->projection = ComputePerspectiveMat4(ToRadian(90.0f), xcb.width / (f32)xcb.height, 0.01f, 100.0f);

    return mem;
}

u32 GetJPEGFiles(const char* dir, LinearAllocator* alloc) {

    auto d = opendir(dir);
    u32 ret = 0;
    if(d) {
        dirent* dir;
        while ((dir = readdir(d)) != NULL) {

            auto strLen = str_len(dir->d_name);
            auto dot = Max((i32)0, (i32)strLen - 5);
            auto fileExt = dir->d_name + dot;

            if(str_cmp(fileExt, ".jpg")) {

                auto path = (const char**)linear_allocate(alloc, sizeof(char) * strLen);
                memcpy(path, dir->d_name, strLen);
                ret++;
            }
        }
        closedir(d);
    }

    return ret;
}

void AppCoro(coroutine* coro, void* arg) {

    auto engine = (EngineState*)arg;

    auto top = engine->alloc.top;
    auto honkSize = ReadFile("../res/honk.png", engine->scratch.base, engine->scratch.cap);
    ASSERT(honkSize != ~u64(0));
    auto honkImg = DecodePNGMemory(engine->scratch.base, honkSize, &engine->alloc);
    honkImg.format = VK_FORMAT_R8G8B8A8_SRGB;

    auto vikingRoomsize = ReadFile("../res/viking_room.png", engine->scratch.base, engine->scratch.cap);
    ASSERT(vikingRoomsize != ~u64(0));
    auto vikingRoomImg = DecodePNGMemory(engine->scratch.base, vikingRoomsize, &engine->alloc);
    vikingRoomImg.format = VK_FORMAT_R8G8B8A8_SRGB;

    auto honk       = CreateVkTexture(&engine->core, &engine->gpu, honkImg);
    auto vikingRoom = CreateVkTexture(&engine->core, &engine->gpu, vikingRoomImg);
    {
        auto transfer = AcquireTransferResources(&engine->exeRes);

        BeginCmdState(&engine->core, &engine->exeRes.cpuCmdAlloc, &transfer);
        auto alloc0 = (byte*)linear_alloc(&engine->gpu.uploadMemory, honk.memory.size);
        auto alloc1 = (byte*)linear_alloc(&engine->gpu.uploadMemory, vikingRoom.memory.size);
        memcpy(alloc0, honkImg.img, honk.memory.size);
        memcpy(alloc1, vikingRoomImg.img, vikingRoom.memory.size);
        PushCPUCommandFreeHost(&transfer, alloc0, honk.memory.size);
        PushCPUCommandFreeHost(&transfer, alloc1, vikingRoom.memory.size);
        engine->alloc.top = top;

        IssueGPUImageBarrier(&engine->core, transfer.cmd, honk, {
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED
            }
        );
        IssueGPUCopytoImage(&engine->core, transfer.cmd, honk, engine->gpu.hostBuffer, alloc0 - (byte*)engine->gpu.uploadMemory.base);
        
        IssueGPUImageBarrier(&engine->core, transfer.cmd, vikingRoom, {
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED
            }
        );
        IssueGPUCopytoImage(&engine->core, transfer.cmd, vikingRoom, engine->gpu.hostBuffer, alloc1 - (byte*)engine->gpu.uploadMemory.base);
        EndCmdState(&engine->core, &transfer);

        auto graphics = AcquireGraphicsResources(&engine->exeRes);
        BeginCmdState(&engine->core, &engine->exeRes.cpuCmdAlloc, &graphics);
        IssueGPUImageBarrier(&engine->core, graphics.cmd, vikingRoom, {
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED
            }
        );
        IssueGPUImageBarrier(&engine->core, graphics.cmd, honk, {
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED
            }
        );
        EndCmdState(&engine->core, &graphics);

        IssueGPUCommands(&engine->core, &engine->gpu, engine->gpu.transferQueue, &transfer, 0, 0, 0, 0, 0);
        IssueGPUCommands(&engine->core, &engine->gpu, engine->gpu.graphicsQueue, &graphics, 0, 0, 0, 0, 0);

        InFlight(&engine->exeRes, &transfer);
        InFlight(&engine->exeRes, &graphics);
    }

    auto honkHandle       = RegisterTexture(&engine->core, &engine->gpu, &engine->exeRes, honk,       engine->exeRes.textureSampler);
    auto vikingRoomHandle = RegisterTexture(&engine->core, &engine->gpu, &engine->exeRes, vikingRoom, engine->exeRes.textureSampler);

    auto quad = (Vertex*)linear_top(&engine->gpu.uploadMemory);
    quad[0].pos = {0, 0, 0};
    quad[0].uv = {0, 1};

    quad[1].pos = {0, 1.f, 0};
    quad[1].uv = {0, 0.f};

    quad[2].pos = {1.f, 1.f, 0};
    quad[2].uv = {1.f, 0.f};

    quad[3].pos = {1.f, 0, 0};
    quad[3].uv = {1.f, 1.f};
    u32* quadIndicies = (u32*)(quad+4);
    quadIndicies[0] = 0;
    quadIndicies[1] = 1;
    quadIndicies[2] = 2;
    quadIndicies[3] = 0;
    quadIndicies[4] = 2;
    quadIndicies[5] = 3;
    LoadedInfo quadInfo;
    quadInfo.indexOffset = (byte*)quadIndicies - engine->gpu.uploadMemory.base;
    quadInfo.indexSize = sizeof(u32) * 6;
    quadInfo.vertexOffset = (byte*)quad - engine->gpu.uploadMemory.base;
    quadInfo.vertexSize = sizeof(Vertex) * 4;
    auto quadModel = UploadModel(&engine->core, &engine->gpu, &engine->exeRes, quadInfo);

    auto roomInfo = LoadOBJ(engine->gpu.uploadMemory.base, (byte*)linear_top(&engine->gpu.uploadMemory), "../res/rooom.obj");
    auto roomModel = UploadModel(&engine->core, &engine->gpu, &engine->exeRes, roomInfo);
    engine->draws = (DrawInfo*)linear_allocate(&engine->alloc, sizeof(DrawInfo) * 10);

    while(engine->xcb_ctx.open) {

        engine->drawCount = 3;
        memset(engine->draws, 0, sizeof(DrawInfo) * 3);
        engine->draws[0].advancePipeline = true;
        engine->draws[0].instanceCount = 0;
        engine->draws[0].model = quadModel;

        engine->draws[1].instanceCount = 3;
        engine->draws[1].model = quadModel;
        engine->draws[2].instanceCount = 1;
        engine->draws[2].model = roomModel;

        engine->instCount = 4;
        memset(engine->instances, 0, sizeof(InstanceInfo) * 4);
        engine->instances[0].textureIndex = honkHandle;
        engine->instances[0].transform = ComputeRotarionXMat4(0);
        engine->instances[0].translation = {0, 0, 0};
        
        engine->instances[1].textureIndex = honkHandle;
        engine->instances[1].transform = ComputeRotarionXMat4(0);
        engine->instances[1].translation = {1.5, 0, 0};

        engine->instances[2].textureIndex = vikingRoomHandle;
        engine->instances[2].transform = ComputeRotarionXMat4(0);
        engine->instances[2].translation = {4, 0, 0};
        engine->instances[3].textureIndex = vikingRoomHandle;
        engine->instances[3].transform = ComputeRotarionYMat4(ToRadian(90.f));
        engine->instances[3].translation = {3.5, 0, 0};

        engine->globalArgs->projectionViewMatrix = engine->projection * LookAt(engine->camera.position, engine->camera.position + engine->camera.direction);

        yield_coroutine(&engine->coro, 5);
    }

    yield_coroutine(&engine->coro, 5);
    DestroyTexture(&engine->core, &engine->gpu, honk);
    DestroyTexture(&engine->core, &engine->gpu, vikingRoom);

    free_gpu_block(&engine->gpu.gpuAllocator, honk.memory);
    free_gpu_block(&engine->gpu.gpuAllocator, vikingRoom.memory);
}

i32 main(i32 argc, const char** argv) {

    auto mem = init_global_state(Megabyte(1), Megabyte(512), 512);
    
    auto xcb_connection = xcb_connect(0, 0);
    if (!xcb_connection || xcb_connection_has_error(xcb_connection)) {
        global_print("sic", "error connecting to X server: ", xcb_connection ? xcb_connection_has_error(xcb_connection) : -1, '\n');
        global_io_flush();
        return 1;
    }
    EngineState state;
    InitEngineState(&state, xcb_connection, mem, Megabyte(512));

    byte stack[KILO_BYTE * 1024];
    init_coroutine(&state.coro, AppCoro, &state, stack + (KILO_BYTE * 1024));

    auto instanceGPUblock = allocate_gpu_block(&state.gpu.gpuAllocator, sizeof(InstanceInfo) * 512, sizeof(InstanceInfo));
    auto begin = std::chrono::high_resolution_clock::now();

    while(state.xcb_ctx.open) {
        ScopedAllocator save(&state.core.vkScratch);

        std::this_thread::sleep_for(milli_second_t(10));
        auto end = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<milli_second_t>(end - begin).count();
        Update(state.connection, &state.keyboard, &state.xcb_ctx, &state, delta);
        begin = std::chrono::high_resolution_clock::now();

        state.exeRes.inFlightCmds = RetireInFlightCmd(&state.core, &state.gpu, &state.exeRes, state.exeRes.inFlightCmds, state.exeRes.cmds);
        if(AreRenderResourcesReady(&state.exeRes) && state.exeRes.inFlightCmds == 0) {

            auto imgAcquired = AcquireResource(&state.exeRes.semaphorePool);
            auto fboImgIndex = IssueSwapChainAcquire(&state.core, &state.gpu, &state.fbo, imgAcquired, nullptr);
            if(fboImgIndex == ~u32(0)) {
                ReleaseResource(&state.exeRes.semaphorePool, imgAcquired);
                RecreateSwapChain(&state.core, &state.gpu, &state.fbo, state.renderPass, state.xcb_ctx.width, state.xcb_ctx.height);
                state.xcb_ctx.width = state.fbo.width;
                state.xcb_ctx.height = state.fbo.height;
                f32 ratio = (f32)state.fbo.width / (f32)state.fbo.height;
                state.projection = ComputePerspectiveMat4(ToRadian(90.0f), ratio, 0.01f, 100.0f);
                continue;
            }

            auto cmd = AcquireGraphicsResources(&state.exeRes);
            BeginCmdState(&state.core, &state.exeRes.cpuCmdAlloc, &cmd);

            auto renderCompleted = AcquireResource(&state.exeRes.semaphorePool);
            auto descriptor = AcquireResource(&state.exeRes.globalRenderParamDescriptors);
            auto cmdSemaphoreRelease = (CommandReleaseSemaphore*)( (byte*)cmd.currentCmd + sizeof(CpuCommand));
            cmdSemaphoreRelease->op = CMD_RELEASE_SEMAPHORE;
            cmdSemaphoreRelease->len = sizeof(VkSemaphore) * 2;
            cmdSemaphoreRelease->semaphores[0] = imgAcquired;
            cmdSemaphoreRelease->semaphores[1] = renderCompleted;

            auto cmdDescRelease = (CommandReleaseDescriptor*)(cmdSemaphoreRelease->semaphores + 2);
            cmdDescRelease->op = CMD_RELEASE_DESCRIPTOR;
            cmdDescRelease->len = sizeof(descriptor) + 10;
            cmdDescRelease->elemSize = sizeof(descriptor);
            cmdDescRelease->descPool = &state.exeRes.globalRenderParamDescriptors;
            memcpy(cmdDescRelease->descriptors, &descriptor, sizeof(descriptor));

            auto cmdAllocFree = (CommandFreeHostAlloc*)(cmdDescRelease->descriptors + sizeof(descriptor));
            cmdAllocFree->op = CMD_FREE_HOST_ALLOC;
            cmdAllocFree->len = 2 * sizeof(Allocation);
            cmdAllocFree->allocs[0].ptr = linear_alloc(&state.gpu.uploadMemory, sizeof(GlobalRenderParams));
            cmdAllocFree->allocs[0].size = sizeof(GlobalRenderParams);
            cmdAllocFree->allocs[1].ptr = linear_alloc(&state.gpu.uploadMemory, sizeof(InstanceInfo) * 100);
            cmdAllocFree->allocs[1].size = sizeof(InstanceInfo) * 100;
            cmd.currentCmd = cmdAllocFree;

            state.globalArgs = (GlobalRenderParams*)(cmdAllocFree->allocs[0].ptr);
            state.instances = (InstanceInfo*)(cmdAllocFree->allocs[1].ptr);

            resume_coroutine(&state.coro);
            FlushDescriptorUpdates(&state.core, &state.gpu, &state.exeRes.descriptorUpdates);
            RecordGPUCopyBarrier(&state.core, &state.gpu, &cmd,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                descriptor.descriptorMemBlocks[0], state.globalArgs
            );
            RecordGPUCopyBarrier(&state.core, &state.gpu, &cmd,
                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                instanceGPUblock, state.instances
            );

            VkDescriptorSet sets[2] = {descriptor.set, state.exeRes.globalTextureDescriptor.set};
            RecordGPUDraw(&state.core, &state.gpu, &state.fbo, &state.program,  &cmd, 2, sets, fboImgIndex, state.drawCount, state.draws, instanceGPUblock.offset);

            EndCmdState(&state.core, &cmd);
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            IssueGPUCommands(&state.core, &state.gpu, state.gpu.graphicsQueue , &cmd, 1, &imgAcquired, &waitStage, 1, &renderCompleted);
            IssueFBOPresent(&state.gpu, &state.fbo, fboImgIndex, renderCompleted);

            InFlight(&state.exeRes, &cmd);
        }

        state.xcb_ctx.keySymbolBuffer.Clear();
        global_io_flush();
    }

    VK_CALL(state.core.vkScratch, vkDeviceWaitIdle, state.gpu.logicalDevice);
    resume_coroutine(&state.coro);

    for(u32 i = 0; i < state.exeRes.inFlightCmds; i++) {
        RetireCmdState(&state.core, &state.gpu, &state.exeRes, state.exeRes.cmds + i);
    }
    DestroyVkExecutionResources(&state.core, &state.gpu, &state.exeRes);
    DestroyVkFbo(&state.core, &state.gpu, &state.fbo);
    VK_CALL(state.core.vkScratch, vkDestroyRenderPass, state.gpu.logicalDevice, state.program.renderPass, &state.core.vkAllocator);
    for(u32 i = 0; i < state.program.subpassCount; i++) {

        for(u32 k = 0; k < state.program.subpasses[i].pipelineCount; k++) {

            auto layout   = state.program.subpasses[i].pipelines[k].layout;
            auto pipeline = state.program.subpasses[i].pipelines[k].pipeline;
            VK_CALL(state.core.vkScratch, vkDestroyPipelineLayout, state.gpu.logicalDevice, layout, &state.core.vkAllocator);
            VK_CALL(state.core.vkScratch, vkDestroyPipeline, state.gpu.logicalDevice, pipeline, &state.core.vkAllocator);
        }
    }

    DestroyVkGPU(&state.core, &state.gpu);
    DestroyVkCore(&state.core);
    xcb_disconnect(state.connection);

    return 0;
}
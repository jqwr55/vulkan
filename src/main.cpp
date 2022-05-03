#include <common.h>

#define VULKAN_DEBUG 1
#include <vulkan.h>
#include <debug.h>
#include <graphics.h>

#include <time.h>
#include <typeinfo>
#include <thread>
#include <pthread.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <tiny_obj_loader.h>
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


void OnDrawFlatRetire(VkRenderContext* ctx, void* resources) {

    auto resourceCount = Mem<u32>(resources);
    byte* it = (byte*)(resources + 4);
    for(u32 i = 0; i < resourceCount; i++) {
        auto res = (DrawflatCmdResources*)it;
        ReleaseResource(&ctx->semaphorePool, res->imgAcquired);
        ReleaseResource(&ctx->semaphorePool, res->renderCompleted);
        ReleaseResource(&ctx->descriptorSetPool, res->descriptor);
        
        for(u32 k = 0; k < res->allocationCount; k++) {
            linear_free(&ctx->uploadMemory, res->allocations[k].ptr, res->allocations[k].size);
        }
        
        it = it + sizeof(DrawflatCmdResources) + res->allocationCount * sizeof(UploadAllocation);
    }
}

void ScreenCapture(EngineState* state, VkRenderContext* ctx, u32* screenShot, u32* counter) {

    /*
    bool mKey = 0;
    if(*screenShot > 60 && mKey) {
        *screenShot = 0;

        char fileName[256];
        
        VkCommandBuffer cmd;
        TryAcquireResource(&ctx->graphicsCmdPool, &cmd);

        VkBeginCmd(&ctx->vkCtx, cmd);
        auto img = ctx->swapChainFrames[0].colorImg;
        VkImageMemoryBarrier imgBarrier{};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.image = img;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgBarrier.subresourceRange.layerCount = 1;
        imgBarrier.subresourceRange.levelCount = 1;
        imgBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        auto srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        auto dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdPipelineBarrier, cmd, srcStage, dstStage, 0,  0, nullptr, 0, nullptr, 1, &imgBarrier);

        VkBufferImageCopy region{};
        region.imageExtent = {ctx->width, ctx->height, 1};
        region.imageOffset = {0,0,0};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.bufferImageHeight = ctx->height;
        region.bufferOffset = ctx->uploadMemory.top;
        region.bufferRowLength = ctx->width;
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdCopyImageToBuffer, cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ctx->hostBuffer, 1, &region);
        VK_CALL(ctx->vkCtx.vkScratch , vkEndCommandBuffer, cmd);

        VkFence fence;
        TryAcquireResource(&ctx->fencePool, &fence);
        VK_CALL(ctx->vkCtx.vkScratch , vkResetFences, ctx->vkCtx.logicalDevice, 1, &fence);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        VK_CALL(ctx->vkCtx.vkScratch , vkQueueSubmit, ctx->graphicsQueue, 1, &submitInfo, fence);
        VK_CALL(ctx->vkCtx.vkScratch , vkWaitForFences, ctx->vkCtx.logicalDevice, 1, &fence, true, ~u64(0));

        ReleaseResource(&ctx->fencePool, fence);
        ReleaseResource(&ctx->graphicsCmdPool, cmd);

        auto texels = (Pixel*)linear_top(&ctx->uploadMemory);
        for(u32 i = 0; i < ctx->width * ctx->height; i++) {
            Swap(&texels[i].r, &texels[i].b);
        }
        local_print((byte*)fileName, 256, "sus", "screen_shot_", (*counter)++, ".png");
        stbi_write_png(fileName, ctx->width, ctx->height, STBI_rgb_alpha, texels, ctx->width * STBI_rgb_alpha);
    }
    */
}
void Update(xcb_connection_t* connection, xcb_context* ctx, EngineState *state, u64 deltaTime) {

    state->time += 0.01;
    state->delta++;

    consume_xcb_events(connection, 1, ctx);

    ComputeCameraVelocity(&state->camera, ctx->keys, 0.0001 * deltaTime);
    state->camera.position = state->camera.position + state->camera.vel;
    state->camera.vel = state->camera.vel * 0.95;

    bool ctrl = true;
    if(!ctrl) {

        f64 cursorX = ctx->cursorsX;
        f64 cursorY = ctx->cursorsY;
        f32 verticalAngle = ((ctx->height * 0.5) - cursorY) / (f32)(ctx->height * 0.5f );
        f32 horizontalAngle = ((ctx->width * 0.5) - cursorX) / (f32)(ctx->width * 0.5f );
        ctx->cursorsX = (f64)ctx->width * 0.5;
        ctx->cursorsY = (f64)ctx->height * 0.5;
        RotateCamera(&state->camera, verticalAngle, -horizontalAngle);

        // xcb_warp_pointer(connection, 0, ctx->window, 0,0,0,0, ctx->cursorsX, ctx->cursorsY);
        // xcb_flush(connection);
    }
}


void* ThreadLoop(void* mem) {

    auto block = (ThreadCommBlock*)mem;
    u32 counter = 0;

    char name[256];
    char localMem[512];
    LinearAllocator io = make_linear_allocator(localMem, 512);

    while(block->run) {

        std::this_thread::sleep_for(milli_second_t(5));

        u32 imgSize = block->x * block->y * sizeof(Pixel);
        auto size = circular_read_size(&block->images, imgSize);
        while(size >= imgSize) {

            auto img = (Pixel*)circular_get_read_ptr(&block->images, imgSize);
            local_print((byte*)name, 256, "sus", "../bin/screen_shoot", counter++, ".png");
            stbi_write_png(name, block->x, block->y, STBI_rgb_alpha, img, block->x * STBI_rgb_alpha);

            VkLog(&io, "ssc", "[general info] screenshot taken ", name, '\n');
            VkFlushLog(&io);

            circular_advance_read(&block->images, imgSize);
            size = circular_read_size(&block->images, imgSize);
        }

    }
}
void global_io_flush_wrapper(void* user, LinearAllocator* io) {

    write(STDOUT_FILENO, io->base, io->top);
    io->top = 0;
}
void OnRetireDummy(VkRenderContext*, void*) {

}
byte* InitState(byte* mem, u32 memSize, xcb_context xcb, EngineState* state, ThreadCommBlock* comms) {

    comms->run = true;
    comms->images = make_ring_buffer(mem, Megabyte(32));
    comms->x = 480;
    comms->y = 640;
    mem += Megabyte(32);

    i32 monitorCount = 0;
    state->camera.position = {2,2,2};
    state->camera.direction = normalize(vec<f32,3>{4, 0, 0} - state->camera.position);
    state->camera.vel = {0,0,0};

    state->threadComm = comms;
    state->time = 0;
    state->delta = 0;
    state->fullscreen = false;
    state->projection = ComputePerspectiveMat4(ToRadian(90.0f), xcb.width / (f32)xcb.height, 0.01f, 100.0f);

    return mem;
}

i32 main(i32 argc, const char** argv) {

    auto mem = init_global_state(0, Megabyte(256), 512);
    auto alloc = make_linear_allocator((byte*)mem, Megabyte(256));
    auto scratch = make_linear_allocator( (byte*)mem + Megabyte(252) , Megabyte(4));
    
    auto honkSize = ReadFile("/media/anon/34214e02-00c7-4b1f-a7b2-e86564e184d22/sources/dev/Projects/vulkan/res/honk.png", scratch.base, scratch.cap);
    ASSERT(honkSize != ~u64(0));
    auto honkImg = MakeImagePNG(scratch.base, &alloc);

    auto vikingRoomsize = ReadFile("/media/anon/34214e02-00c7-4b1f-a7b2-e86564e184d22/sources/dev/Projects/vulkan/res/viking_room.png", scratch.base, scratch.cap);
    ASSERT(vikingRoomsize != ~u64(0));
    auto vikingRoomImg = MakeImagePNG(scratch.base, &alloc);
   
    auto generalSize = ReadFile("/home/anon/Desktop/banan.jpeg", scratch.base, scratch.cap);
    ASSERT(generalSize != ~u64(0));
    auto generalImg = MakeJPEGImage(scratch.base, generalSize, &alloc);

    xcb_connection_t* c = xcb_connect(0, 0);
    xcb_context xcb_ctx[2];
    auto id0 = xcb_generate_id(c);
    //auto id1 = xcb_generate_id(c);
    xcb_ctx[0] = make_xcb_context(c, id0, 640, 480, "vk_render_window");
    //xcb_ctx[1] = make_xcb_context(c, id1, 640, 480, "misc_window");

    VkCoreContext core;
    VkGPUContext gpu;
    VkProgram program;
    VkFbo fbo;
    VkExecutionResources exeRes;

    VkRenderContextConfig config;
    config.logMask =    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   | 
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    config.windowHeight          = 480;
    config.windowWidth           = 640;
    config.ioLogBufferSize       = Kilobyte(2);
    config.scractchSize          = Kilobyte(64);
    config.vkHeapSize            = Megabyte(32);
    config.uploadHeapSize        = Megabyte(32);
    config.localHeapSize         = Kilobyte(4);
    config.gpuHeapSize           = Megabyte(64);
    config.gpuhHeapMaxAllocCount = 256;
    config.fboCount              = 3;

    MakeVkCoreContext(&core, config, {}, &alloc);
    VkXcbSurfaceCreateInfoKHR xcbSurfaceCreateInfo{};
    xcbSurfaceCreateInfo.connection = c;
    xcbSurfaceCreateInfo.window = xcb_ctx->window;
    xcbSurfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    VkSurfaceKHR surface;
    VK_CALL(core.vkScratch, vkCreateXcbSurfaceKHR, core.vkInstance, &xcbSurfaceCreateInfo, &core.vkAllocator, &surface);

    VkPhysicalDevice gpus[2];
    auto gpuCount = GetGPUs(&core, gpus);
    auto selectedGPU = PickPhysicalDevice(&core, surface, gpuCount, gpus, DEVICE_EXTENSIONS, SIZE_OF_ARRAY(DEVICE_EXTENSIONS));
    gpu = MakeVkGPUContext(&core, selectedGPU, surface, config, {}, &alloc);
    exeRes = MakeVkExecutionResources(&core, &gpu, {}, &alloc);

    VkPrgoramDescriptor programDescription{};
    programDescription.width = xcb_ctx[0].width;
    programDescription.height = xcb_ctx[0].height;
    programDescription.depthFormat = FindDepthFormat(&core, gpu.device);
    programDescription.colorFormat = GetSurfaceFormat(&core, gpu.device, surface).format;
    
    program = MakeVKProgram(&core, &gpu, programDescription, {}, &alloc);
    fbo = MakeVkFbo(&core, &gpu, program.renderPass, surface, c, xcb_ctx, {}, &alloc);

    ThreadCommBlock comms;
    EngineState state;
    mem = InitState((byte*)linear_allocator_top(&alloc), linear_allocator_free_size(&alloc), xcb_ctx[0], &state, &comms);
    xcb_warp_pointer(c, 0, xcb_ctx[0].window, 0,0,0,0, xcb_ctx[0].width/2, xcb_ctx[0].height/2);
    xcb_flush(c);

    auto honk       = CreateVkTexture(&core, &gpu, honkImg);
    auto vikingRoom = CreateVkTexture(&core, &gpu, vikingRoomImg);
    auto general    = CreateVkTexture(&core, &gpu, generalImg);

    CmdState cmds[2];
    u32 inFlightCmds = 0;
    {
        auto transfer = AcquireTransferResources(&exeRes);
        auto graphics = AcquireGraphicsResources(&exeRes);
        BeginCmdState(&core, &exeRes.cpuCmdBuffer, &graphics);
        BeginCmdState(&core, &exeRes.cpuCmdBuffer, &transfer);

        RecordVkTextureUpload(&core, &gpu, &transfer, &graphics, honk, honkImg);
        RecordVkTextureUpload(&core, &gpu, &transfer, &graphics, vikingRoom, vikingRoomImg);
        RecordVkTextureUpload(&core, &gpu, &transfer, &graphics, general, generalImg);

        EndCmdState(&core, &transfer);
        EndCmdState(&core, &graphics);

        IssueCmdState(&core, &gpu, gpu.transferQueue, &transfer, 0, 0, 0, 0, 0);
        IssueCmdState(&core, &gpu, gpu.graphicsQueue, &graphics, 0, 0, 0, 0, 0);

        cmds[0] = transfer;
        cmds[1] = graphics;
        inFlightCmds = 2;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto honkHandle = RegisterTexture(&exeRes, honk, program.textureSampler, program.textureDescriptors);
    auto generalHandle = RegisterTexture(&exeRes, general, program.textureSampler, program.textureDescriptors);
    auto vikingRoomHandle = RegisterTexture(&exeRes, vikingRoom, program.textureSampler, program.textureDescriptors);

    auto quad = (Vertex*)linear_top(&gpu.uploadMemory);
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
    quadInfo.indexOffset = (byte*)quadIndicies - gpu.uploadMemory.base;
    quadInfo.indexSize = sizeof(u32) * 6;
    quadInfo.vertexOffset = (byte*)quad - gpu.uploadMemory.base;
    quadInfo.vertexSize = sizeof(Vertex) * 4;
    auto quadModel = UploadModel(&core, &gpu, &exeRes, quadInfo);

    auto roomInfo = LoadOBJ(gpu.uploadMemory.base, (byte*)linear_top(&gpu.uploadMemory), "../res/rooom.obj");
    auto roomModel = UploadModel(&core, &gpu, &exeRes, roomInfo);

    auto instanceGPUblock = allocate_gpu_block(&gpu.gpuAllocator, sizeof(InstanceInfo) * 512, sizeof(InstanceInfo));

    DrawInfo draws[2];
    draws[0].instanceCount = 3;
    draws[0].model = quadModel;
    draws[1].instanceCount = 1;
    draws[1].model = roomModel;

    u32 screenShot = 0;
    auto begin = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    while(xcb_ctx[0].open || xcb_ctx[1].open) {
        ScopedAllocator save(&core.vkScratch);

        screenShot++;
        std::this_thread::sleep_for(milli_second_t(20));
        auto end = std::chrono::high_resolution_clock::now();
        Update(c, xcb_ctx, &state, std::chrono::duration_cast<milli_second_t>(end - begin).count());
        begin = std::chrono::high_resolution_clock::now();

        bool mKey = 0 && screenShot > 10;

        inFlightCmds = RetireInFlightCmd(&core, &gpu, &program, &exeRes, inFlightCmds, cmds);
        FlushDescriptorUpdates(&core, &gpu, &exeRes.descriptorUpdates);

        if(AreRenderResourcesReady(&program, &exeRes) && inFlightCmds == 0) {
          
            auto imgAcquired = AcquireResource(&exeRes.semaphorePool);
            auto fboImgIndex = IssueSwapChainAcquire(&core, &gpu, &fbo, imgAcquired, nullptr);
            if(fboImgIndex == ~u32(0)) {
                ReleaseResource(&exeRes.semaphorePool, imgAcquired);
                RecreateSwapChain(&core, &gpu, &fbo, program.renderPass, xcb_ctx[0].width, xcb_ctx[0].height);
                xcb_ctx[0].width = fbo.width;
                xcb_ctx[0].height = fbo.height;
                continue;
            }

            auto cmd = AcquireGraphicsResources(&exeRes);
            BeginCmdState(&core, &exeRes.cpuCmdBuffer, &cmd);

            auto renderCompleted = AcquireResource(&exeRes.semaphorePool);
            auto descriptor = AcquireResource(&program.descriptorSetPool);
            auto cmdSemaphoreRelease = (CmdReleaseSemaphore*)( (byte*)cmd.currentCmd + sizeof(CpuCmd));
            cmdSemaphoreRelease->op = CMD_RELEASE_SEMAPHORE;
            cmdSemaphoreRelease->len = sizeof(VkSemaphore) * 2;
            cmdSemaphoreRelease->semaphores[0] = imgAcquired;
            cmdSemaphoreRelease->semaphores[1] = renderCompleted;
            auto cmdDescRelease = (CmdReleaseDescriptor*)(cmdSemaphoreRelease->semaphores + 2);
            cmdDescRelease->op = CMD_RELEASE_DESCRIPTOR;
            cmdDescRelease->len = sizeof(Descriptor);
            cmdDescRelease->descriptors[0] = descriptor;
            auto cmdAllocFree = (CmdFreeHostAlloc*)(cmdDescRelease->descriptors + 1);
            cmdAllocFree->op = CMD_FREE_HOST_ALLOC;
            cmdAllocFree->len = 2 * sizeof(Allocation);
            cmdAllocFree->allocs[0].ptr = linear_alloc(&gpu.uploadMemory, sizeof(CommonParams));
            cmdAllocFree->allocs[0].size = sizeof(CommonParams);
            cmdAllocFree->allocs[1].ptr = linear_alloc(&gpu.uploadMemory, sizeof(InstanceInfo) * 100);
            cmdAllocFree->allocs[1].size = sizeof(InstanceInfo) * 100;
            cmd.currentCmd = cmdAllocFree;

            auto renderArgs = (CommonParams*)(cmdAllocFree->allocs[0].ptr);
            auto instances = (InstanceInfo*)(cmdAllocFree->allocs[1].ptr);
            renderArgs->projectionViewMatrix = state.projection * LookAt(state.camera.position, state.camera.position + state.camera.direction);

            instances[0].textureIndex = honkHandle;
            instances[0].transform = ComputeRotarionXMat4(0);
            instances[0].translation = {0, 0, 0};
            instances[1].textureIndex = generalHandle;
            instances[1].transform = ComputeRotarionXMat4(0);
            instances[1].translation = {1.5, 0, 0};
            instances[2].textureIndex = vikingRoomHandle;
            instances[2].transform = ComputeRotarionXMat4(0);
            instances[2].translation = {4, 0, 0};
            instances[3].textureIndex = vikingRoomHandle;
            instances[3].transform = ComputeRotarionYMat4(ToRadian(90.f));
            instances[3].translation = {3.5, 0, 0};

            RecordCopyBarrier(&core, &gpu, &cmd, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, {descriptor.offset, sizeof(CommonParams)}, renderArgs);
            RecordCopyBarrier(&core, &gpu, &cmd, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, instanceGPUblock, instances);
            RecordDraw(&core, &gpu, &fbo, &program, &cmd, descriptor.set, fboImgIndex, 2, draws, instanceGPUblock.offset);

            EndCmdState(&core, &cmd);
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            IssueCmdState(&core, &gpu, gpu.graphicsQueue , &cmd, 1, &imgAcquired, &waitStage, 1, &renderCompleted);
            IssuePresentImg(&gpu, &fbo, fboImgIndex, renderCompleted);
            cmds[inFlightCmds++] = cmd;
        }

        global_io_flush();
    }

    comms.run = false;

    vkDeviceWaitIdle(gpu.logicalDevice);
    DestroyTexture(&core, &gpu, honk);
    DestroyTexture(&core, &gpu, vikingRoom);
    DestroyTexture(&core, &gpu, general);

    free_gpu_block(&gpu.gpuAllocator, general.memory);
    free_gpu_block(&gpu.gpuAllocator, honk.memory);
    free_gpu_block(&gpu.gpuAllocator, vikingRoom.memory);

    for(u32 i = 0; i < inFlightCmds; i++) {
        RetireCmdState(&core, &gpu, &program, &exeRes, cmds + i);
    }
    DestroyVkExecutionResources(&core, &gpu, &exeRes);
    DestroyVkFbo(&core, &gpu, &fbo);
    DestroyVkProgram(&core, &gpu, &program);

    DestroyVkGPU(&core, &gpu);
    DestroyVkCore(&core);

    xcb_disconnect(c);

    return 0;
}

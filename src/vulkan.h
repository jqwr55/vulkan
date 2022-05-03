#pragma once

#include <graphics.h>
#include <atomic>

#include <window.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xcb.h>

extern const char* VALIDATION_LAYERS[1];
extern const char* DEVICE_EXTENSIONS[7];
extern const char* INSTANCE_EXTENSIONS[3];

struct Vertex {
    vec<f32,3> pos;
    vec<f32,2> uv;
};
struct CommonParams {

    Mat4<f32> projectionViewMatrix;
    Mat4<f32> inverseProjectionViewMatrix;
    vec<f32,4> viewDir;
    vec<f32,4> viewPos;
    vec<f32,4> viewRight;
    vec<f32,4> viewUp;
    f32 time;
    void* mem;
    void* host;
    void* device;
};
template<typename T> struct ResourcePool {
    T* resources;
    u32 resourceCount;
};
template<typename T> struct ResourcePoolAtomic {
    std::atomic<u32> semaphore;
    u32 top;
    T* begin;
};
struct QueueFamilies {
    u32 graphicsFamily;
    u32 computeFamily;
    u32 transferFamily;
    u32 presentFamily;
};
struct VkTextureInfo {
    VkImage img;
    VkImageView view;
    MemBlock memory;
};
struct SwapChainFrame {
    VkImage colorImg;
    VkImageView colorImgView;
    VkTextureInfo depthImg;
    VkImageView depthImgView;
    VkFramebuffer frameBuffer;
};
struct Descriptor {
    VkDescriptorSet set;
    u64 offset;
};


struct InstanceInfo{
    Mat<f32, 3,3> transform;
    vec<f32, 3> translation;
    u32 textureIndex;
};
struct PendingDescriptorUpdates {
    VkWriteDescriptorSet write[10];
    union {
        VkDescriptorImageInfo imgInfo;
        VkDescriptorBufferInfo bufferInfo;
    } infos[10];
    u32 count;
};

struct VkContext {
    VkAllocationCallbacks       vkAllocator;
    VkInstance                  vkInstance;
    VkDebugUtilsMessengerEXT    vkDebugMessenger;
    VkPhysicalDevice            device;
    VkDevice                    logicalDevice;

    LocalMallocState vkHeap;
    LinearAllocator vkScratch;
};
struct VkCoreContext {

    VkAllocationCallbacks       vkAllocator;
    VkInstance                  vkInstance;
    VkDebugUtilsMessengerEXT    vkDebugMessenger;

    LocalMallocState vkHeap;
    LinearAllocator vkScratch;
    LinearAllocator ioLog;
    bit_mask16 logSeverity;
};
struct VkFbo {

    SwapChainFrame* frames;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;

    u32 frameCount;
    u32 height;
    u32 width;
    VkFormat fboColorFormat;
    VkFormat depthFormat;
};
struct VkGPUContext {

    VkPhysicalDevice device;
    VkDevice         logicalDevice;

    VkBuffer         hostBuffer;
    VkBuffer         deviceBuffer;
    VkDeviceMemory   hostMemory;
    VkDeviceMemory   deviceMemory;

    VkQueue          graphicsQueue;
    VkQueue          presentQueue;
    VkQueue          computeQueue;
    VkQueue          transferQueue;
    QueueFamilies    families;

    VkCommandPool    commandPoolGraphics;
    VkCommandPool    commandPoolTransfer;
    VkDescriptorPool descriptorPool;

    CoalescingLinearAllocator uploadMemory;
    GpuHeap gpuAllocator;

    u64 hostMemoryDeviceAddress;
    u64 deviceMemoryDeviceAddress;
    u32 textureDescriptorCount;
};
struct SubpassDescriptor {

    u32 count;
    VkDescriptorBindingFlags* flags;
    VkDescriptorSetLayoutBinding* layoutBindings;
};
struct VkPrgoramDescriptor {

    SubpassDescriptor* subpasses;
    VkFormat depthFormat;
    VkFormat colorFormat;
    u32 width;
    u32 height;
    u32 subpassCount;
};
struct ProgramSubPass {

    u32 descriptorSetCount;
    VkDescriptorSetLayout* descriptorLayouts;
    VkDescriptorSet* descriptorSets;
    VkPipelineLayout pipelineLayout;
};
struct VkProgram {

    VkDescriptorSet          textureDescriptors;
    VkPipelineLayout         pipelineLayout;
    VkRenderPass             renderPass;
    VkPipeline               graphicsPipeline;
    VkDescriptorSetLayout    UBOdescriptorLayout;
    VkDescriptorSetLayout    textureDescriptorLayout;
    ResourcePool<Descriptor> descriptorSetPool;
    VkSampler                textureSampler;
    VkPipelineBindPoint      bindpoint;

    u32 subpassCount;
    ProgramSubPass* subpasses;
};
struct VkExecutionResources {

    CircularAllocator cpuCmdBuffer;
    ResourcePool<VkSemaphore> semaphorePool;
    ResourcePool<VkFence> fencePool;
    ResourcePool<VkCommandBuffer> transferCmdPool;
    ResourcePool<VkCommandBuffer> graphicsCmdPool;
    PendingDescriptorUpdates descriptorUpdates;

    u16 head;
    u16 textureSlotTable[512];
};

enum CpuCMDOp : u8 {

    CMD_EOB,
    CMD_FREE_HOST_ALLOC,
    CMD_RELEASE_SEMAPHORE,
    CMD_RELEASE_DESCRIPTOR,
};
struct __attribute__ ((packed)) CpuCmd {
    CpuCMDOp op;
    u16 len;
};
struct Allocation {
    void* ptr;
    u32 size;
};
struct __attribute__ ((packed))CmdFreeHostAlloc {
    CpuCMDOp op;
    u16 len;
    Allocation allocs[];
};
struct __attribute__ ((packed)) CmdReleaseSemaphore {
    CpuCMDOp op;
    u16 len;
    VkSemaphore semaphores[];
};
struct __attribute__ ((packed)) CmdReleaseDescriptor {
    CpuCMDOp op;
    u16 len;
    Descriptor descriptors[];
};

struct VkRenderContext {

    VkContext vkCtx;

    VkFormat swapChainImageFormat;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;

    VkDescriptorSetLayout UBOdescriptorLayout;
    VkDescriptorSetLayout textureDescriptorLayout;
    VkPipelineLayout pipelineLayout;
    VkRenderPass renderPass;
    VkPipeline graphicsPipeline;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;
    VkQueue transferQueue;
    QueueFamilies families;

    VkCommandPool commandPoolGraphics;
    VkCommandPool commandPoolTransfer;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet textureDescriptors;
    VkSampler textureSampler;

    VkBuffer hostBuffer;
    VkBuffer deviceBuffer;
    VkDeviceMemory hostMemory;
    VkDeviceMemory deviceMemory;
    u64 hostMemoryDeviceAddress;
    u64 deviceMemoryDeviceAddress;
    u32 textureDescriptorCount;

    StaticBufferLocalMalloc<SwapChainFrame> swapChainFrames;
    ResourcePool<VkSemaphore> semaphorePool;
    ResourcePool<VkFence> fencePool;
    ResourcePool<VkCommandBuffer> transferCmdPool;
    ResourcePool<VkCommandBuffer> graphicsCmdPool;
    ResourcePool<Descriptor> descriptorSetPool;
    PendingDescriptorUpdates descriptorUpdates;

    const char* title;
    LinearAllocator ioLogBuffer;
    CoalescingLinearAllocator uploadMemory;
    LocalMallocState localHeap;
    GpuHeap gpuAllocator;

    u32 height;
    u32 width;
    bit_mask16 logSeverity;

    u16 head;
    u16 textureSlotTable[512];
};

struct ThreadCommBlock {

    bool run;
    RingBuffer images;
    u32 x;
    u32 y;
    VkRenderContext* ctx;
};

struct EngineState {
    ThreadCommBlock* threadComm;
    Camera camera;
    Mat4<f32> projection;
    f32 time;
    u32 delta;
    bool fullscreen;
};

struct ModelDesciption {
    u32 vertexOffset;
    u32 vertexCount;
    u32 indexOffset;
    u32 indexCount;
};

struct LoadedInfo {
    u32 vertexOffset;
    u32 vertexSize;
    u32 indexOffset;
    u32 indexSize;
};
struct DrawInfo {
    ModelDesciption model;
    u32 instanceCount;
};

struct CmdState {
    VkCommandBuffer cmd;
    VkFence complete;

    void* cmdsOnRetrire;
    void* currentCmd;
    ResourcePool<VkCommandBuffer>* cmdSource;
};
struct UploadAllocation {
    void* ptr;
    u32 size;
};
struct DrawflatCmdResources {
    VkSemaphore imgAcquired;
    VkSemaphore renderCompleted;
    Descriptor descriptor;
    EngineState* engine;
    u32 allocationCount;
    UploadAllocation allocations[];
};
struct SwapChainResult {
    VkSwapchainKHR swapChain;
    VkFormat format;
    vec<u32, 2> dims;
};

struct VkRenderContextConfig {
    

    u32 windowHeight;
    u32 windowWidth;
    u32 fboCount;

    u32 scractchSize;
    u32 vkHeapSize;
    u32 ioLogBufferSize;
    u32 uploadHeapSize;
    u32 localHeapSize;
    u32 gpuHeapSize;
    u32 gpuhHeapMaxAllocCount;

    bit_mask16 logMask;
};

template<typename T> T AcquireResource(ResourcePool<T>* pool) {
    ASSERT(pool->resourceCount != 0);
    pool->resourceCount -= pool->resourceCount != 0;
    return pool->resources[pool->resourceCount];
}
template<typename T> bool TryAcquireResource(ResourcePool<T>* pool, T* res) {

    bool nonEmpty = pool->resourceCount != 0;
    pool->resourceCount -= nonEmpty;
    *res = pool->resources[pool->resourceCount];
    return nonEmpty;
}
template<typename T> bool IsResourceAvailable(ResourcePool<T>* pool) {
    return pool->resourceCount != 0;
}


template<typename T> bool TryAcquireResourceAtomic(ResourcePoolAtomic<T>* pool, T* res) {

    while(pool->semaphore.compare_exchange_strong(0, 1));

    bool nonEmpty = pool->top != 0;
    pool->top -= nonEmpty;
    *res = pool->resources[pool->top];

    pool->semaphore--;

    return nonEmpty;
}
template<typename T> void ReleaseResourceAtomic(ResourcePoolAtomic<T>* pool, T resource) {

    while(pool->semaphore.compare_exchange_strong(0, 1));
    pool->resources[pool->top++] = resource;
    pool->semaphore--;
}

template<typename T> void ReleaseResource(ResourcePool<T>* pool, T resource) {
    pool->resources[pool->resourceCount++] = resource;
}

void VkFlushLog(LinearAllocator* ioBuffer);
void VkLog(LinearAllocator* ioBuffer, const char* format, ...);
void* vkLocalMalloc(void* user, size_t size, size_t align, VkSystemAllocationScope scope);
void vkLocalFree(void* user, void* mem);
void* vkLocalRealloc(void* user, void* og, size_t size, size_t alignment, VkSystemAllocationScope scope);
bool CheckValidationLayerSupport(const char** validationLayersRequired, u32 count);
u32 GetRequiredExtensions(const char** ext, const char** instanceExtensions, u32 instanceExtensionsCount);
VkBool32 VKAPI_PTR DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
VkDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT(VkContext* ctx, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo);
void DestroyDebugUtilsMessengerEXT(VkContext* ctx, VkDebugUtilsMessengerEXT debugMessenger);
f32 DeviceScore(VkContext* ctx, VkPhysicalDevice device);
void PrintQueueFamilyFlags(VkQueueFlags flags);
void PrintQueueFamilies(VkContext* ctx, VkPhysicalDevice device);
QueueFamilies GetQueueFamilies(VkContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface);
void PrintAvailableDeviceExt(VkContext* ctx, VkPhysicalDevice device);
bool CheckDeviceExtensionSupport(VkContext* ctx, VkPhysicalDevice device, const char** deviceExtensions, u32 deviceExtensionsCount);
void PrintDeviceProperties(VkContext* ctx, VkPhysicalDevice device);
bool IsDeviceSuitable(VkContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface, const char** requiredDeviceExt, u32 count);
u32 GetGPUs(VkCoreContext* core, VkPhysicalDevice* dst);
VkPhysicalDevice PickPhysicalDevice(VkCoreContext* ctx, VkSurfaceKHR surface, u32 gpuCount, VkPhysicalDevice* gpus, const char** requiredDeviceExt, u32 count);
VkDevice CreateLogicalDevice(VkContext* ctx, QueueFamilies families, VkPhysicalDevice physicalDevice);
VkSurfaceFormatKHR ChooseSwapSurfaceFormat(VkSurfaceFormatKHR* availableFormats, u32 formatCount);
VkPresentModeKHR ChooseSwapPresentMode(VkPresentModeKHR* availablePresentModes, u32 count, VkPresentModeKHR* preferredOrder);
VkExtent2D ChooseSwapExtent(VkSurfaceCapabilitiesKHR capabilities, u32 width, u32 height);
SwapChainResult CreateSwapChain(VkCoreContext* ctx, u32 frameCount, VkPhysicalDevice device, VkDevice logicalDevice, VkSurfaceKHR surface, vec<u32,2> dims, QueueFamilies families, VkSwapchainKHR oldChain);
VkFormat FindSupportedFormat(VkContext* ctx, u32 count, VkFormat* candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
VkFormat FindDepthFormat(VkCoreContext* ctx, VkPhysicalDevice device);
VkSurfaceFormatKHR GetSurfaceFormat(VkCoreContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface);
bool hasStencilComponent(VkFormat format);
VkImage CreateImg2D(VkCoreContext* ctx, VkDevice logicalDevice, vec<u32,2> dims, u32* families, u32 familyCount, VkFormat format, VkImageUsageFlags usage);
VkTextureInfo UploadVkTexture(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* exeRes, ImageDescriptor img);
u32 CreateSwapChainFrames(VkCoreContext* core, VkGPUContext* gpu, SwapChainFrame* dst, VkRenderPass renderPass, VkSwapchainKHR swapChain, vec<u32, 2> dim, VkFormat depthFormat, VkFormat colorFormat);
VkShaderModule CreateShaderModule(VkCoreContext* ctx, VkDevice logicalDevice, const char* source, u32 len);
VkDescriptorSetLayout CreateDescriptorSetLayouts(VkCoreContext* ctx, VkDevice logicalDevice, VkDescriptorSetLayoutBinding* bindings, VkDescriptorBindingFlags* flags, u32 count);
VkPipelineLayout CreatePipelineLayout(VkContext* ctx, VkDescriptorSetLayout* descriptorSetLayout, u32 count);
VkRenderPass CreateRenderPass(VkContext* ctx, VkFormat swapChainFormat);
VkPipeline CreateGraphicsPipeline(VkCoreContext* ctx, VkDevice logicalDevice, VkPipelineLayout layout, VkRenderPass renderPass, u32 width, u32 height);
void CreateFramebuffers(VkRenderContext* ctx, VkRenderPass renderPass, SwapChainFrame* dst, u32 frameCount);
void CreateCommandPool(VkRenderContext* ctx);
u32 RegisterTexture(VkExecutionResources* ctx, VkTextureInfo texture, VkSampler sampler, VkDescriptorSet set);
ModelDesciption UploadModel(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, LoadedInfo model);
void DestroyRenderContext(VkRenderContext* ctx);
MemBlock BackImgMemory(VkCoreContext* ctx, VkDevice logicalDevice, VkDeviceMemory memory, GpuHeap* gpuAllocator, VkImage img);
void RetireCmdState(VkCoreContext* ctx, VkGPUContext* gpu, VkProgram* program, VkExecutionResources* res, CmdState* cmdState);
void DestroyTexture(VkCoreContext* core, VkGPUContext* gpu, VkTextureInfo texture);
void IssueCmdState(VkCoreContext* ctx, VkGPUContext* gpu, VkQueue queue,  CmdState* cmd, u32 waitCount, VkSemaphore* wait, VkPipelineStageFlags* stages, u32 signalCount, VkSemaphore* signal);
void IssuePresentImg(VkGPUContext* gpu, VkFbo* fbo, u32 imgIndex, VkSemaphore wait);
void Render(EngineState* state, VkRenderContext* ctx, CmdState cmd, u32 fboImg, MemBlock block, DrawflatCmdResources* resources, DrawInfo* draws, u32 drawCount, bool screenShot);
u32 MakeVkRenderContext(VkRenderContext* ctx, xcb_connection_t* connection, xcb_context xcb, byte* memory, u32 memorySize, VkRenderContextConfig config);
void BeginCmdState(VkCoreContext* core, CircularAllocator* alloc, CmdState* cmd);
void RecreateSwapChain(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkRenderPass renderPass, u32 width, u32 height);
void EndCmdState(VkCoreContext* core, CmdState* cmd);
CmdState AcquireGraphicsResources(VkExecutionResources* res);
CmdState AcquireTransferResources(VkExecutionResources* res);
bool AreRenderResourcesReady(VkProgram* program, VkExecutionResources* res);
void EndCmdState(VkCoreContext* core, CmdState* cmd);
u32 RetireInFlightCmd(VkCoreContext* core, VkGPUContext* gpu, VkProgram* program, VkExecutionResources* res, u32 count, CmdState* cmds);
u32 IssueSwapChainAcquire(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkSemaphore signalSemaphore, VkFence signalFence);
void RecordVkTextureUpload(VkCoreContext* core, VkGPUContext* gpu, CmdState* transfer, CmdState* graphics, VkTextureInfo vkTex, ImageDescriptor img);
VkTextureInfo CreateVkTexture(VkCoreContext* core, VkGPUContext* gpu, ImageDescriptor img);
MemBlock BackImgMemory(VkCoreContext* ctx, VkDevice logicalDevice, VkDeviceMemory memory, GpuHeap* gpuAllocator, VkImage img);
void FlushDescriptorUpdates(VkCoreContext* core, VkGPUContext* gpu, PendingDescriptorUpdates* pending);
void RecordDraw(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkProgram* program, CmdState* cmdState, VkDescriptorSet descriptors, u32 imgIndex, u32 drawCount, DrawInfo* draws, u64 instanceOffset);
void RecordCopyBarrier(VkCoreContext* core, VkGPUContext* gpu, CmdState* cmdState, VkAccessFlags dstMask, VkAccessFlags dstStage, MemBlock dst, void* src);
void ExecuteCpuCmds(VkGPUContext* gpu, VkProgram* program, VkExecutionResources* exeRes, byte* cmdsBegin);

void MakeVkCoreContext(VkCoreContext* dst, VkRenderContextConfig config, Printer printer, LinearAllocator* alloc);
VkGPUContext MakeVkGPUContext(VkCoreContext* core, VkPhysicalDevice gpu, VkSurfaceKHR surface, VkRenderContextConfig config, Printer printer, LinearAllocator* alloc);
VkExecutionResources MakeVkExecutionResources(VkCoreContext* core, VkGPUContext* gpu, Printer printer, LinearAllocator* alloc);
VkFbo MakeVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkRenderPass renderPass, VkSurfaceKHR surface, xcb_connection_t* connection, xcb_context* xcb_ctx, Printer printer, LinearAllocator* alloc);
VkProgram MakeVKProgram(VkCoreContext* core, VkGPUContext* gpu, VkPrgoramDescriptor description, Printer printer, LinearAllocator* alloc);

void DestroyVkCore(VkCoreContext* core);
void DestroyVkGPU(VkCoreContext* core, VkGPUContext* gpu);
void DestroyVkExecutionResources(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res);
void DestroyVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo);
void DestroyVkProgram(VkCoreContext* core, VkGPUContext* gpu, VkProgram* program);
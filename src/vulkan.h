#pragma once

#include <math3d.h>
#include <atomic>

#include "window.h"
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xcb.h>

extern const char* VALIDATION_LAYERS[1];
extern const char* DEVICE_EXTENSIONS[6];
extern const char* INSTANCE_EXTENSIONS[4];

struct Vertex {
    vec<f32,3> pos;
    vec<f32,2> uv;
};
struct Vertex2 {
    vec<f32,2> pos;
    u32 col;
};
struct GlobalRenderParams {

    Mat4<f32> projectionViewMatrix;
    Mat4<f32> inverseProjectionViewMatrix;
    vec<f32,4> viewDir;
    vec<f32,4> viewPos;
    vec<f32,4> viewRight;
    vec<f32,4> viewUp;
    f32 time;
};
template<typename T>
struct ResourcePool {
    T* resources;
    u32 resourceCount;
};
template<typename T>
struct ResourcePoolAtomic {
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
    u32 width;
    u32 height;
};
struct SwapChainFrame {
    VkImage colorImg;
    VkImageView colorImgView;
    VkTextureInfo depthImg;
    VkImageView depthImgView;
    VkFramebuffer frameBuffer;
};

template<u32 DESCRIPTOR_COUNT>
struct DescriptorSet {
    VkDescriptorSet set;
    MemBlock descriptorMemBlocks[DESCRIPTOR_COUNT];
};


struct InstanceInfo{
    Mat<f32, 3,3> transform;
    vec<f32, 3> translation;
    u32 textureIndex;
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
    bool advancePipeline;
    bool advanceSubpass;
};
struct ImageBarrierArgs {
    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    VkAccessFlags flushCache;
    VkAccessFlags invalidateCache;
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    u32 srcQueue;
    u32 dstQueue;
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
    u64              totalMemorySize;

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
struct VertexDescriptor {

    u32 attribCount;
    u32 bindingCount;

    VkVertexInputAttributeDescription* attribs;
    VkVertexInputBindingDescription* bindings;
};
struct DescriptorSetLayoutInfo {
    
    u32 bindingCount;
    u32* bindingMemorySizes;
    VkDescriptorBindingFlags* flags;
    VkDescriptorSetLayoutBinding* layoutBindings;
};
struct PipelineDescriptor {

    VkPipelineBindPoint bindpoint;
    StaticBuffer<const byte> fragByteCode;
    StaticBuffer<const byte> vertByteCode;

    VertexDescriptor attribDescriptor;
    VkPipelineInputAssemblyStateCreateInfo inputAsm;
    VkPipelineTessellationStateCreateInfo  tessellationState;
    VkPipelineViewportStateCreateInfo      viewportState;
    VkPipelineRasterizationStateCreateInfo rasterizationState;
    VkPipelineMultisampleStateCreateInfo   multisampleState;
    VkPipelineDepthStencilStateCreateInfo  depthStencilState;
    VkPipelineColorBlendStateCreateInfo    colorBlendState;
    VkPipelineDynamicStateCreateInfo       dynamicState;
};
struct PipelineInfo {
    VkPipeline       pipeline;
    VkPipelineLayout layout;
};
struct PipelineSubpass {

    u32 pipelineCount;
    VkPipelineBindPoint bindpoint;
    PipelineInfo pipelines[];
};
struct VkProgram2 {

    u32              subpassCount;
    VkRenderPass     renderPass;
    PipelineSubpass* subpasses;
};
struct VkProgram {

    VkPipelineLayout         pipelineLayout;
    VkRenderPass             renderPass;
    VkPipeline               graphicsPipeline0;
    VkPipelineBindPoint      bindpoint;

    VkDescriptorSetLayout UBOdescriptorLayout;
    VkDescriptorSetLayout textureDescriptorLayout;

    ResourcePool<DescriptorSet<1>> descriptorSetPool;
    VkDescriptorSet textureDescriptors;
};
struct VkExecutionResources {

    CircularAllocator cpuCmdAlloc;
    ResourcePool<VkSemaphore> semaphorePool;
    ResourcePool<VkFence> fencePool;
    ResourcePool<VkCommandBuffer> transferCmdPool;
    ResourcePool<VkCommandBuffer> graphicsCmdPool;

    CmdState* cmds;
    u32 inFlightCmds;

    CmdState* transferPending;
    CmdState* graphicsPending;

    ResourcePool<DescriptorSet<1>> globalRenderParamDescriptors;
    DescriptorSet<1> globalTextureDescriptor;
    PendingDescriptorUpdates descriptorUpdates;

    VkDescriptorSetLayout layout0;
    VkDescriptorSetLayout layout1;

    VkSampler textureSampler;

    u16 head;
    u16 textureSlotTable[512];
};

enum CpuCMDOp : u8 {

    CMD_EOB,
    CMD_FREE_HOST_ALLOC,
    CMD_RELEASE_SEMAPHORE,
    CMD_RELEASE_DESCRIPTOR,
};
struct __attribute__ ((packed)) CpuCommand {
    CpuCMDOp op;
    u16 len;
};
struct Allocation {
    void* ptr;
    u32 size;
};
struct __attribute__ ((packed)) CommandFreeHostAlloc {
    CpuCMDOp op;
    u16 len;
    Allocation allocs[];
};
struct __attribute__ ((packed)) CommandReleaseSemaphore {
    CpuCMDOp op;
    u16 len;
    VkSemaphore semaphores[];
};
struct __attribute__ ((packed)) CommandReleaseDescriptor {
    CpuCMDOp op;
    u16 len;
    u16 elemSize;
    void* descPool;
    byte descriptors[];
};

struct SwapChainResult {
    VkSwapchainKHR swapChain;
    VkFormat format;
    vec<u32, 2> dims;
};
struct VkRenderContextConfig {
    

    u32 windowHeight;
    u32 windowWidth;

    u32 scractchSize;
    u32 vkHeapSize;
    u32 ioLogBufferSize;
    u32 uploadHeapSize;
    u32 gpuHeapSize;
    u32 gpuhHeapMaxAllocCount;

    bit_mask16 logMask;
};

template<typename T>
T AcquireResource(ResourcePool<T>* pool) {
    ASSERT(pool->resourceCount != 0);
    pool->resourceCount -= pool->resourceCount != 0;
    return pool->resources[pool->resourceCount];
}
template<typename T>
bool TryAcquireResource(ResourcePool<T>* pool, T* res) {

    bool nonEmpty = pool->resourceCount != 0;
    pool->resourceCount -= nonEmpty;
    *res = pool->resources[pool->resourceCount];
    return nonEmpty;
}
template<typename T>
bool IsResourceAvailable(ResourcePool<T>* pool) {
    return pool->resourceCount != 0;
}


template<typename T>
bool TryAcquireResourceAtomic(ResourcePoolAtomic<T>* pool, T* res) {

    while(pool->semaphore.compare_exchange_strong(0, 1));

    bool nonEmpty = pool->top != 0;
    pool->top -= nonEmpty;
    *res = pool->resources[pool->top];

    pool->semaphore--;

    return nonEmpty;
}
template<typename T>
void ReleaseResourceAtomic(ResourcePoolAtomic<T>* pool, T resource) {

    while(pool->semaphore.compare_exchange_strong(0, 1));
    pool->resources[pool->top++] = resource;
    pool->semaphore--;
}

template<typename T>
void ReleaseResource(ResourcePool<T>* pool, T resource) {
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
void PrintAvailableDeviceExt(VkPhysicalDevice device);
bool CheckDeviceExtensionSupport(VkContext* ctx, VkPhysicalDevice device, const char** deviceExtensions, u32 deviceExtensionsCount);
void PrintDeviceProperties(VkPhysicalDevice device);
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
VkShaderModule CreateShaderModule(VkCoreContext* ctx, VkDevice logicalDevice, const byte* source, u32 len);
VkDescriptorSetLayout CreateDescriptorSetLayouts(VkCoreContext* ctx, VkDevice logicalDevice, VkDescriptorSetLayoutBinding* bindings, VkDescriptorBindingFlags* flags, u32 count);
VkPipelineLayout CreatePipelineLayout(VkContext* ctx, VkDescriptorSetLayout* descriptorSetLayout, u32 count);
VkRenderPass CreateRenderPass(VkCoreContext* ctx, VkDevice logicalDevice, VkFormat colorFormat, VkFormat depthFormat);
VkPipeline CreateGraphicsPipeline(VkCoreContext* ctx, VkDevice logicalDevice, VkPipelineLayout layout, VkRenderPass renderPass, VkPipelineVertexInputStateCreateInfo* input, const char* vertexPath, const char* fragPath, u32 width, u32 height);

u32 UnRegisterTexture(VkExecutionResources* ctx, u32 handle);
u32 RegisterTexture(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* ctx, VkTextureInfo texture, VkSampler sampler);
ModelDesciption UploadModel(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, LoadedInfo model);

MemBlock BackImgMemory(VkCoreContext* ctx, VkDevice logicalDevice, VkDeviceMemory memory, GpuHeap* gpuAllocator, VkImage img);
void RetireCmdState(VkCoreContext* ctx, VkGPUContext* gpu, VkExecutionResources* res, CmdState* cmdState);
void DestroyTexture(VkCoreContext* core, VkGPUContext* gpu, VkTextureInfo texture);
void IssueGPUCommands(VkCoreContext* ctx, VkGPUContext* gpu, VkQueue queue,  CmdState* cmd, u32 waitCount, VkSemaphore* wait, VkPipelineStageFlags* stages, u32 signalCount, VkSemaphore* signal);
void IssueFBOPresent(VkGPUContext* gpu, VkFbo* fbo, u32 imgIndex, VkSemaphore wait);
void IssueGPUCopytoImage(VkCoreContext* core, VkCommandBuffer commandBuffer, VkTextureInfo dst, VkBuffer srcBuffer, u64 srcOffset);

void BeginCmdState(VkCoreContext* core, CircularAllocator* alloc, CmdState* cmd);
void RecreateSwapChain(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkRenderPass renderPass, u32 width, u32 height);
void EndCmdState(VkCoreContext* core, CmdState* cmd);
CmdState AcquireGraphicsResources(VkExecutionResources* res);
CmdState AcquireTransferResources(VkExecutionResources* res);
bool AreRenderResourcesReady(VkExecutionResources* res);
void EndCmdState(VkCoreContext* core, CmdState* cmd);
u32 RetireInFlightCmd(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, u32 count, CmdState* cmds);
u32 IssueSwapChainAcquire(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkSemaphore signalSemaphore, VkFence signalFence);
void RecordGPUTextureUpload(VkCoreContext* core, VkGPUContext* gpu, CmdState* transfer, CmdState* graphics, VkTextureInfo vkTex, ImageDescriptor img);
VkTextureInfo CreateVkTexture(VkCoreContext* core, VkGPUContext* gpu, ImageDescriptor img);
MemBlock BackImgMemory(VkCoreContext* ctx, VkDevice logicalDevice, VkDeviceMemory memory, GpuHeap* gpuAllocator, VkImage img);
void FlushDescriptorUpdates(VkCoreContext* core, VkGPUContext* gpu, PendingDescriptorUpdates* pending);

void RecordGPUDrawDebug(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkProgram* program, CmdState* cmdState, u32 imgIndex, u32 setCount, VkDescriptorSet* descriptors, u32 drawCount, DrawInfo* draws, u64 instanceOffset);
void RecordGPUDraw(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkProgram2* program, CmdState* cmdState, u32 setCount, VkDescriptorSet* descriptors , u32 imgIndex, u32 drawCount, DrawInfo* draws, u64 instanceOffset);
void IssueGPUImageBarrier(VkCoreContext* core, VkCommandBuffer commandBuffer, VkTextureInfo img, ImageBarrierArgs args);
void RecordGPUCopyBarrier(VkCoreContext* core, VkGPUContext* gpu, CmdState* cmdState, VkAccessFlags dstMask, VkAccessFlags dstStage, MemBlock dst, void* src);
void ExecuteCpuCmds(VkGPUContext* gpu, VkExecutionResources* exeRes, byte* cmdsBegin);

void InFlight(VkExecutionResources* exe, CmdState* cmd);
void PushCPUCommandFreeHost(CmdState* state, void* mem, u32 size);

void MakeVkCoreContext(VkCoreContext* dst, VkRenderContextConfig config, Printer printer, LinearAllocator* alloc);
VkGPUContext MakeVkGPUContext(VkCoreContext* core, VkPhysicalDevice gpu, VkSurfaceKHR surface, VkRenderContextConfig config, Printer printer, LinearAllocator* alloc);
VkExecutionResources MakeVkExecutionResources(VkCoreContext* core, VkGPUContext* gpu, Printer printer, LinearAllocator* alloc);
VkFbo MakeVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkRenderPass renderPass, VkSurfaceKHR surface, u32 frameCount, xcb_connection_t* connection, xcb_context* xcb_ctx, Printer printer, LinearAllocator* alloc);
VkProgram MakeVkProgram(VkCoreContext* core, VkGPUContext* gpu, VkRenderPass renderPass, Printer printer, LinearAllocator* alloc);
VkDescriptorSetLayout MakeDescriptorSets(VkCoreContext* core, VkGPUContext* gpu, const DescriptorSetLayoutInfo* info, u32 setCount, LinearAllocator* alloc);
u32 ComputeDescriptorSize(const DescriptorSetLayoutInfo* info);
PipelineInfo MakePipeline(VkCoreContext* core, VkGPUContext* gpu, u32 subpassIndex, VkRenderPass renderPass, u32 descLayoutCount, VkDescriptorSetLayout* descLayouts, const PipelineDescriptor* pipeDescription);
void PushAttribute(VertexDescriptor* dst, u32 location, u32 binding, VkFormat format, u32 offset,  LinearAllocator* alloc);
void PushAttributeBinding(VertexDescriptor* dst, u32 binding, u32 stride, VkVertexInputRate inputRate, LinearAllocator* alloc);

void DestroyVkCore(VkCoreContext* core);
void DestroyVkGPU(VkCoreContext* core, VkGPUContext* gpu);
void DestroyVkExecutionResources(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res);
void DestroyVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo);
void DestroyVkProgram(VkCoreContext* core, VkGPUContext* gpu, VkProgram* program);
const char* GetVkResultEnumStr(VkResult result);


extern const DescriptorSetLayoutInfo global_render_info_descriptor_layout_info;
extern const DescriptorSetLayoutInfo global_textures_descriptor_layout_info;
extern const PipelineDescriptor global_debug_flat2d_pipeline;
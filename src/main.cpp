#define VULKAN_DEBUG 1

#include <glm/gtx/euler_angles.hpp>

#include <common.h>
#include <debug.h>
#include <graphics.h>

#include <time.h>
#include <typeinfo>
#include <thread>
#include <atomic>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <tiny_obj_loader.h>
#include <pthread.h>

const char* validationLayers[] = {
    "VK_LAYER_KHRONOS_validation",
};
const char* deviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_DEVICE_GROUP_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_MAINTENANCE3_EXTENSION_NAME,
};
const char* instanceExtensions[] = {
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME,
};
constexpr u32 MAX_FRAMES_IN_FLIGHT = 1;
enum KEY_MASK : u32 {
    KEY_NONE        = 0,
    KEY_W           = 1 << 0,
    KEY_A           = 1 << 1,
    KEY_S           = 1 << 2,
    KEY_D           = 1 << 3,
    KEY_SPACE       = 1 << 4,
    KEY_LEFT_SHIFT  = 1 << 5,
};
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
struct VkContext {
    VkAllocationCallbacks       vkAllocator;
    VkInstance                  vkInstance;
    VkDebugUtilsMessengerEXT    vkDebugMessenger;
    VkPhysicalDevice            device;
    VkDevice                    logicalDevice;

    LocalMallocState vkHeap;
    LinearAllocator vkScratch;
};

template<typename T> struct ResourcePool {
    T* resources;
    u32 resourceCount;
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

template<typename T> struct ResourcePoolAtomic {
    std::atomic<u32> semaphore;
    u32 top;
    T* begin;
};
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

struct RenderContext {
    
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

    DynamicBufferLocal<SwapChainFrame, MultiAllocPool<8>, pool_multi_allocate<8>, pool_multi_free<8>> swapChainFrames;
    ResourcePool<VkSemaphore> semaphorePool;
    ResourcePool<VkFence> fencePool;
    ResourcePool<VkCommandBuffer> transferCmdPool;
    ResourcePool<VkCommandBuffer> graphicsCmdPool;
    ResourcePool<Descriptor> descriptorSetPool;
    PendingDescriptorUpdates descriptorUpdates;

    GLFWwindow* glfwHandle;
    const char* title;
    MultiAllocPool<sizeof(void*)> pointerPool;
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
};

struct EngineState {
    GLFWmonitor* primary;
    RenderContext* ctx;
    ThreadCommBlock* threadComm;
    Camera camera;
    Mat4<f32> projection;
    f32 time;
    u32 delta;
    bool fullscreen;
};



void VkFlushLog(LinearAllocator* ioBuffer) {
    write(STDOUT_FILENO, ioBuffer->base, ioBuffer->top);
    ioBuffer->top = 0;
}
void VkLog(LinearAllocator* ioBuffer, const char* format, ...) {

    auto t = time(nullptr);
    auto date = localtime(&t);
    date->tm_year += 1900;
    // y:m:d h:m:s
    again:
    auto save = ioBuffer->top;

    auto end = local_print((byte*)linear_allocator_top(ioBuffer), linear_allocator_free_size(ioBuffer), "c icici c icici s", '[', date->tm_year, ':', date->tm_mon, ':', date->tm_mday, ' ', date->tm_hour, ':', date->tm_min, ':', date->tm_sec, "] ");
    ioBuffer->top = end - ioBuffer->base;

    if(ioBuffer->top == ioBuffer->cap) {
        ioBuffer->top = save;
        VkFlushLog(ioBuffer);
        goto again;
    }

    va_list args;
    va_start(args, format);
    end = print_fn_v((byte*)linear_allocator_top(ioBuffer), linear_allocator_free_size(ioBuffer), format, args);
    va_end(args);

    ioBuffer->top = end - ioBuffer->base;
    if(ioBuffer->top == ioBuffer->cap) {
        ioBuffer->top = save;
        VkFlushLog(ioBuffer);
        goto again;
    }
}


void* vkLocalMalloc(void* user, size_t size, size_t align, VkSystemAllocationScope scope) {

    auto ctx = (VkContext*)user;
    if(scope == VK_SYSTEM_ALLOCATION_SCOPE_COMMAND) {
        auto mem = linear_aligned_allocate(&ctx->vkScratch, size, align);
        ASSERT(mem);
        return mem;
    }
    auto mem = local_malloc(&ctx->vkHeap, size);
    ASSERT(mem);
    return mem;
}
void vkLocalFree(void* user, void* mem) {

    ASSERT(user);
    auto ctx = (VkContext*)user;
    if(!mem || (byte*)mem <= ctx->vkScratch.base + ctx->vkScratch.cap) {
        return;
    }
    u64 size = local_malloc_allocation_size(mem);
    local_free(&ctx->vkHeap, mem);
}
void* vkLocalRealloc(void* user, void* og, size_t size, size_t alignment, VkSystemAllocationScope scope) {
    
    if(scope == VK_SYSTEM_ALLOCATION_SCOPE_COMMAND) {
        ASSERT(false);
    }
    auto ctx = (VkContext*)user;
    auto fresh = local_malloc(&ctx->vkHeap, size);
    if(!og) {
        local_free(&ctx->vkHeap, og);
        return fresh;
    }

    auto prevSize = local_malloc_allocation_size(og);
    memcpy(fresh, og, prevSize);
    local_free(&ctx->vkHeap, og);

    return fresh;
}

bool CheckValidationLayerSupport(const char** validationLayersRequired, u32 count) {

    u32 layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    VkLayerProperties availableLayers[layerCount];
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

    for(u32 i = 0; i < count; i++) {
        auto name = validationLayersRequired[i];
        bool found = false;

        for(u32 i = 0; i < layerCount; i++) {
            if(str_cmp(name, availableLayers[i].layerName)) {
                found = true;
                break;
            }
        }
        if(!found) {
            return false;
        }
    }
    return true;
}
u32 GetRequiredExtensions(const char** ext, const char** instanceExtensions, u32 instanceExtensionsCount) {

    u32 glfwExtensionCount = 0;
    const char** glfwExtensions;
    GLFW_CALL(glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount));
    memcpy(ext, glfwExtensions, glfwExtensionCount * sizeof(const char*));

    for(u32 i = 0; i < instanceExtensionsCount; i++) {
        ext[glfwExtensionCount++] = instanceExtensions[i];
    }
    if constexpr (ENABLE_VALIDATION_LAYER) {
        ext[glfwExtensionCount++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    return glfwExtensionCount;
}


VkBool32 VKAPI_PTR DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {

    constexpr const char* SEVERITY_STR[] = {
        "VERBOSE",
        "INFO",
        "WARNING",
        "ERROR"
    };
    auto ctx = (RenderContext*)pUserData;
    if(ctx->logSeverity & messageSeverity) {
        VkLog(&ctx->ioLogBuffer, "sscsc", "[vulkan validation layer info] severity ", SEVERITY_STR[(i32)f32_log(messageSeverity, 16)], ' ', pCallbackData->pMessage, '\n');
        VkFlushLog(&ctx->ioLogBuffer);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT(VkContext* ctx, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo) {

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(ctx->vkInstance, "vkCreateDebugUtilsMessengerEXT");
    ASSERT(func);

    VkDebugUtilsMessengerEXT pDebugMessenger;
    VK_CALL(ctx->vkScratch, func, ctx->vkInstance, pCreateInfo, &ctx->vkAllocator, &pDebugMessenger);

    return pDebugMessenger;
}
void DestroyDebugUtilsMessengerEXT(VkContext* ctx, VkDebugUtilsMessengerEXT debugMessenger) {
    
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(ctx->vkInstance, "vkDestroyDebugUtilsMessengerEXT");
    ASSERT(func);
    VK_CALL(ctx->vkScratch, func, ctx->vkInstance, debugMessenger, &ctx->vkAllocator);
}

f32 DeviceScore(VkContext* ctx, VkPhysicalDevice device) {

    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;

    VkPhysicalDeviceBufferDeviceAddressFeatures addressFeature{};
    addressFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    addressFeature.pNext = &indexingFeatures;

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &addressFeature;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceFeatures2, device, &deviceFeatures2);

    VkPhysicalDeviceProperties deviceProperties;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceProperties, device, &deviceProperties);

    auto bindlessSupport = indexingFeatures.descriptorBindingPartiallyBound && indexingFeatures.runtimeDescriptorArray;
    f32 score = 0;
    score += addressFeature.bufferDeviceAddress * 10;
    score += deviceFeatures2.features.geometryShader ? 10 : 0;
    score += deviceProperties.limits.maxImageDimension2D;
    score += bindlessSupport * 10;
    score *= deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
    return score;
}

void PrintQueueFamilyFlags(VkQueueFlags flags) {

    const char* QUEUE_FAMILY_FLAG_STR[] = {
        "VK_QUEUE_GRAPHICS_BIT",
        "VK_QUEUE_COMPUTE_BIT",
        "VK_QUEUE_TRANSFER_BIT",
        "VK_QUEUE_SPARSE_BINDING_BIT",
        "VK_QUEUE_PROTECTED_BIT",
    };

    for(u32 i = 0; i < SIZE_OF_ARRAY(QUEUE_FAMILY_FLAG_STR); i++) {

        u32 mask = (1 << i);
        if( (flags & mask) == mask) {
            global_print("ss", QUEUE_FAMILY_FLAG_STR[i], ", ");
        }
    }
}
void PrintQueueFamilies(VkContext* ctx, VkPhysicalDevice device) {

    u32 queueFamilyCount = 0;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceQueueFamilyProperties, device, &queueFamilyCount, nullptr);
    VkQueueFamilyProperties queueFamilies[queueFamilyCount];
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceQueueFamilyProperties, device, &queueFamilyCount, queueFamilies);

    for(u32 i = 0; i < queueFamilyCount; i++) {

        global_print("susus", "queue family [", i, "] queue count [", queueFamilies[i].queueCount, "] ");
        PrintQueueFamilyFlags(queueFamilies[i].queueFlags);
        global_print("c", '\n');
    }
}

QueueFamilies GetQueueFamilies(VkContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface) {

    u32 queueFamilyCount = 0;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceQueueFamilyProperties, device, &queueFamilyCount, nullptr);
    VkQueueFamilyProperties queueFamilies[queueFamilyCount];
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceQueueFamilyProperties, device, &queueFamilyCount, queueFamilies);

    QueueFamilies families;
    families.graphicsFamily = ~u32(0);
    families.computeFamily  = ~u32(0);
    families.transferFamily = ~u32(0);
    families.presentFamily  = ~u32(0);

    for(u32 i = 0; i < queueFamilyCount; i++) {
        if(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            families.graphicsFamily = i;
            break;
        }
    }
    for(u32 i = 0; i < queueFamilyCount; i++) {
        if( (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            families.computeFamily = i;
            break;
        }
    }
    if(families.computeFamily == ~u32(0)) {
        families.computeFamily = families.graphicsFamily;
    }
    for(u32 i = 0; i < queueFamilyCount; i++) {
        if( (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            families.transferFamily = i;
            break;
        }
    }
    if(families.transferFamily == ~u32(0)) {
        families.transferFamily = families.computeFamily;
    }
    for(u32 i = 0; i < queueFamilyCount; i++) {
        VkBool32 presentSupport = false;
        VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceSupportKHR, device, i, surface, &presentSupport);
        if(presentSupport) {
            families.presentFamily = i;
            break;
        }
    }

    return families;
}

void PrintAvailableDeviceExt(VkContext* ctx, VkPhysicalDevice device) {

    u32 extensionCount;
    VK_CALL(ctx->vkScratch, vkEnumerateDeviceExtensionProperties, device, nullptr, &extensionCount, nullptr);
    VkExtensionProperties availableExtensions[extensionCount];
    VK_CALL(ctx->vkScratch, vkEnumerateDeviceExtensionProperties, device, nullptr, &extensionCount, availableExtensions);

    for(u32 i = 0; i < extensionCount; i++) {
        global_print("sc", availableExtensions[i].extensionName, '\n');
    }
    global_io_flush();
}

bool CheckDeviceExtensionSupport(VkContext* ctx, VkPhysicalDevice device, const char** deviceExtensions, u32 deviceExtensionsCount) {
 
    u32 extensionCount;
    VK_CALL(ctx->vkScratch, vkEnumerateDeviceExtensionProperties, device, nullptr, &extensionCount, nullptr);
    VkExtensionProperties availableExtensions[extensionCount];
    VK_CALL(ctx->vkScratch, vkEnumerateDeviceExtensionProperties, device, nullptr, &extensionCount, availableExtensions);

    for(u32 i = 0; i < deviceExtensionsCount; i++) {
        bool found = false;
        for(u32 k = 0; k < extensionCount; k++) {
            if(str_cmp(deviceExtensions[i], availableExtensions[k].extensionName)) {
                found = true;
            }
        }
        if(!found) return false;
    }

    return true;
}

void PrintDeviceProperties(VkContext* ctx, VkPhysicalDevice device) {

    VkPhysicalDeviceProperties physicalDeviceProperties;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceProperties, device , &physicalDeviceProperties);
    global_print("suc", "apiVersion", physicalDeviceProperties.apiVersion, '\n');
    global_print("suc", "deviceID", physicalDeviceProperties.deviceID, '\n');
    global_print("ssc", "deviceName", physicalDeviceProperties.deviceName, '\n');
    global_print("suc", "deviceType", physicalDeviceProperties.deviceType, '\n');
    global_print("suc", "driverVersion", physicalDeviceProperties.driverVersion, '\n');
    global_print("suc", "pipelineCacheUUID", physicalDeviceProperties.pipelineCacheUUID, '\n');
    global_print("suc", "vendorID", physicalDeviceProperties.vendorID, '\n');
    global_io_flush();
}
bool IsDeviceSuitable(VkContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface, const char** requiredDeviceExt, u32 count) {
    
    auto family = GetQueueFamilies(ctx, device, surface);
    auto extSupport = CheckDeviceExtensionSupport(ctx, device, requiredDeviceExt, count);

    if(!family.graphicsFamily == ~u32(0) || family.presentFamily == ~u32(0) || !extSupport) return false;

    u32 formatCount;
    u32 presentModeCount;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, nullptr);
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfacePresentModesKHR, device, surface, &presentModeCount, nullptr);

    return formatCount && presentModeCount;
}
VkPhysicalDevice PickPhysicalDevice(VkContext* ctx, VkInstance instance, VkSurfaceKHR surface, const char** requiredDeviceExt, u32 count) {

    u32 deviceCount = 0;
    VK_CALL(ctx->vkScratch, vkEnumeratePhysicalDevices, instance, &deviceCount, nullptr);

    if(deviceCount == 0) {
        return nullptr;
    }

    VkPhysicalDevice devices[deviceCount];
    VK_CALL(ctx->vkScratch, vkEnumeratePhysicalDevices, instance, &deviceCount, devices);

    f32 max = 0;
    u32 device = ~u32(0);
    for(u32 i = 0; i < deviceCount; i++) {
        if(!IsDeviceSuitable(ctx, devices[i], surface, requiredDeviceExt, count)) continue;
        f32 score = DeviceScore(ctx, devices[i]);
        device = score > max ? i : device;
        max = score > max ? score : max;
    }
    ASSERT(device != ~u32(0));

    return devices[device];
}



VkDevice CreateLogicalDevice(VkContext* ctx, QueueFamilies families, VkPhysicalDevice physicalDevice) {

    u32 uniqueFamilies[4];
    memcpy(uniqueFamilies, &families, sizeof(u32) * 4);
    u32 uniqueFamilyCount = get_unique(uniqueFamilies, 4);
    VkDeviceQueueCreateInfo queueCreateInfos[uniqueFamilyCount];

    f32 queuePriority = 1.0f;
    for(u32 i = 0; i < uniqueFamilyCount; i++) {
        queueCreateInfos[i] = {};
        queueCreateInfos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[i].queueFamilyIndex = uniqueFamilies[i];
        queueCreateInfos[i].queueCount = 1;
        queueCreateInfos[i].pQueuePriorities = &queuePriority;
    }
    
    VkPhysicalDeviceBufferDeviceAddressFeatures addressFeature{};
    addressFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    addressFeature.bufferDeviceAddress = true;

    VkPhysicalDeviceDescriptorIndexingFeatures bindlessDescriptor{};
    bindlessDescriptor.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
    bindlessDescriptor.pNext = &addressFeature;
    bindlessDescriptor.descriptorBindingPartiallyBound = VK_TRUE;
    bindlessDescriptor.runtimeDescriptorArray = VK_TRUE;
    bindlessDescriptor.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    bindlessDescriptor.descriptorBindingVariableDescriptorCount = VK_TRUE;

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &bindlessDescriptor;
    deviceFeatures2.features.shaderFloat64 = true;
    deviceFeatures2.features.shaderInt64 = true;
    deviceFeatures2.features.shaderInt16 = true;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos;
    createInfo.queueCreateInfoCount = uniqueFamilyCount;
    createInfo.pNext = &deviceFeatures2;;
    
    createInfo.ppEnabledExtensionNames = deviceExtensions;
    createInfo.enabledExtensionCount = SIZE_OF_ARRAY(deviceExtensions);

    if constexpr (ENABLE_VALIDATION_LAYER) {
        createInfo.enabledLayerCount = SIZE_OF_ARRAY(validationLayers);
        createInfo.ppEnabledLayerNames = validationLayers;
    }
    
    VkDevice device;
    VK_CALL(ctx->vkScratch, vkCreateDevice, physicalDevice, &createInfo, &ctx->vkAllocator, &device);

    return device;
}

VkSurfaceFormatKHR ChooseSwapSurfaceFormat(VkSurfaceFormatKHR* availableFormats, u32 formatCount) {
    
    for (u32 i = 0; i < formatCount; i++) {
        if (availableFormats[i].format == VK_FORMAT_B8G8R8A8_SRGB && availableFormats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormats[i];
        }
    }
    return availableFormats[0];
}
VkPresentModeKHR ChooseSwapPresentMode(VkPresentModeKHR* availablePresentModes, u32 count, VkPresentModeKHR* preferredOrder) {
    

    for(u32 i = 0; i < 6; i++) {
        for (u32 k = 0; k < count; k++) {
            if (availablePresentModes[k] == preferredOrder[i]) {
                return availablePresentModes[k];
            }
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D ChooseSwapExtent(VkSurfaceCapabilitiesKHR capabilities, u32 width, u32 height) {
    
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    else {

        VkExtent2D actualExtent = {width, height};
        actualExtent.width = Clamp(actualExtent.width, capabilities.maxImageExtent.width, capabilities.minImageExtent.width);
        actualExtent.height = Clamp(actualExtent.height, capabilities.maxImageExtent.height, capabilities.minImageExtent.height);
        return actualExtent;
    }
}

struct SwapChainResult {
    VkSwapchainKHR swapChain;
    VkFormat format;
};

SwapChainResult CreateSwapChain(VkContext* ctx, vec<u32,2> dims, VkPhysicalDevice device, VkDevice logicalDevice, VkSurfaceKHR surface, QueueFamilies families, VkSwapchainKHR oldChain) {

    u32 formatCount;
    u32 presentModeCount;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, nullptr);
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfacePresentModesKHR, device, surface, &presentModeCount, nullptr);

    VkSurfaceFormatKHR formats[formatCount];
    VkPresentModeKHR presentModes[presentModeCount];
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, formats);
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfacePresentModesKHR, device, surface, &presentModeCount, presentModes);

    VkSurfaceCapabilitiesKHR capabilities;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceCapabilitiesKHR, device, surface, &capabilities);

    VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(formats, formatCount);

    VkPresentModeKHR rank[] = {
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_FIFO_RELAXED_KHR,
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR,
        VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
        VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR,
    };
    VkPresentModeKHR presentMode = ChooseSwapPresentMode(presentModes, presentModeCount, rank);
    VkExtent2D extent = ChooseSwapExtent(capabilities, dims.x, dims.y);
    
    u32 imageCount = capabilities.minImageCount;
    if(capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    SwapChainResult ret;
    ret.format = surfaceFormat.format;

    VkSwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainCreateInfo.surface = surface;
    swapChainCreateInfo.minImageCount = imageCount;
    swapChainCreateInfo.imageFormat = surfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = extent;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;


    u32 queueFamilyIndices[] = {families.graphicsFamily, families.presentFamily};
    if (families.graphicsFamily != families.presentFamily) {
        swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapChainCreateInfo.queueFamilyIndexCount = 2;
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapChainCreateInfo.queueFamilyIndexCount = 0;
        swapChainCreateInfo.pQueueFamilyIndices = nullptr;
    }

    swapChainCreateInfo.preTransform = capabilities.currentTransform;
    swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainCreateInfo.presentMode = presentMode;
    swapChainCreateInfo.clipped = VK_TRUE;
    swapChainCreateInfo.oldSwapchain = oldChain;

    ret.swapChain;
    VK_CALL(ctx->vkScratch, vkCreateSwapchainKHR, logicalDevice, &swapChainCreateInfo, &ctx->vkAllocator, &ret.swapChain);
    return ret;
}
VkFormat FindSupportedFormat(VkContext* ctx, u32 count, VkFormat* candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

    for (u32 i = 0; i < count; i++) {
        VkFormatProperties props;
        VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceFormatProperties, ctx->device, candidates[i], &props);

        bool cond = (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) ||
                    (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features);
        if(cond) {
            return candidates[i];
        }
    }

    return VkFormat{};
}
VkFormat FindDepthFormat(VkContext* ctx) {
    VkFormat formats[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    return FindSupportedFormat(ctx, SIZE_OF_ARRAY(formats), formats, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}


VkImage CreateImg2D(RenderContext* ctx, vec<u32,2> dims, VkFormat format, VkImageUsageFlags usage) {

    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.extent.width = dims.x;
    imgInfo.extent.height = dims.y;
    imgInfo.extent.depth = 1;
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = format;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = usage;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;

    u32 families[] = {ctx->families.graphicsFamily, ctx->families.transferFamily};
    imgInfo.pQueueFamilyIndices = families;
    imgInfo.queueFamilyIndexCount = 2;

    VkImage img;
    VK_CALL(ctx->vkCtx.vkScratch, vkCreateImage, ctx->vkCtx.logicalDevice, &imgInfo, &ctx->vkCtx.vkAllocator, &img);
    return img;
}

MemBlock BackImgMemory(VkContext* ctx, VkDeviceMemory memory, GpuHeap* gpuAllocator, VkImage img) {

    VkMemoryRequirements memRequirements;
    VK_CALL(ctx->vkScratch, vkGetImageMemoryRequirements, ctx->logicalDevice, img, &memRequirements);

    auto block = allocate_gpu_block(gpuAllocator, memRequirements.size, memRequirements.alignment);
    VK_CALL(ctx->vkScratch, vkBindImageMemory, ctx->logicalDevice, img, memory, block.offset);

    return block;
}

VkTextureInfo UploadVkTexture(RenderContext* ctx, ImageDescriptor img) {

    u32 imgSize = img.width * img.height * 4;
    auto texels = img.img;

    auto dst = (Pixel*)linear_top(&ctx->uploadMemory);
    memcpy(dst, texels, imgSize);

    VkTextureInfo ret;
    ret.img = CreateImg2D(ctx, {img.width,img.height}, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ret.memory = BackImgMemory(&ctx->vkCtx, ctx->deviceMemory, &ctx->gpuAllocator, ret.img);

    VkCommandBuffer cmd;
    ASSERT(TryAcquireResource(&ctx->transferCmdPool, &cmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(ctx->vkCtx.vkScratch, vkBeginCommandBuffer, cmd, &begin);

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkImageMemoryBarrier toTransferDstOptimal{};
    toTransferDstOptimal.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDstOptimal.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDstOptimal.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDstOptimal.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstOptimal.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstOptimal.image = ret.img;
    toTransferDstOptimal.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferDstOptimal.subresourceRange.levelCount = 1;
    toTransferDstOptimal.subresourceRange.layerCount = 1;
    toTransferDstOptimal.srcAccessMask = 0;
    toTransferDstOptimal.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdPipelineBarrier, cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &toTransferDstOptimal);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {img.width,img.height, 1};
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdCopyBufferToImage, cmd, ctx->hostBuffer, ret.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VK_CALL(ctx->vkCtx.vkScratch, vkEndCommandBuffer, cmd);

    VkCommandBuffer ownership;
    ASSERT(TryAcquireResource(&ctx->graphicsCmdPool, &ownership));
    VK_CALL(ctx->vkCtx.vkScratch, vkBeginCommandBuffer, ownership, &begin);

    VkImageMemoryBarrier toShaderReadOPt{};
    toShaderReadOPt.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toShaderReadOPt.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShaderReadOPt.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShaderReadOPt.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShaderReadOPt.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShaderReadOPt.image = ret.img;
    toShaderReadOPt.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toShaderReadOPt.subresourceRange.baseMipLevel = 0;
    toShaderReadOPt.subresourceRange.levelCount = 1;
    toShaderReadOPt.subresourceRange.baseArrayLayer = 0;
    toShaderReadOPt.subresourceRange.layerCount = 1;
    toShaderReadOPt.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShaderReadOPt.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdPipelineBarrier, ownership, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &toShaderReadOPt);
    VK_CALL(ctx->vkCtx.vkScratch, vkEndCommandBuffer, ownership);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence;
    ASSERT(TryAcquireResource(&ctx->fencePool, &fence));
    VK_CALL(ctx->vkCtx.vkScratch, vkResetFences, ctx->vkCtx.logicalDevice, 1, &fence);
    VK_CALL(ctx->vkCtx.vkScratch, vkQueueSubmit, ctx->transferQueue, 1, &submitInfo, fence);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = ret.img;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    VK_CALL(ctx->vkCtx.vkScratch, vkCreateImageView, ctx->vkCtx.logicalDevice, &viewInfo, &ctx->vkCtx.vkAllocator, &ret.view);

    VK_CALL(ctx->vkCtx.vkScratch, vkWaitForFences, ctx->vkCtx.logicalDevice, 1, &fence, true, ~u64(0));

    VkSubmitInfo submitInfo2{};
    submitInfo2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo2.commandBufferCount = 1;
    submitInfo2.pCommandBuffers = &ownership;
    VK_CALL(ctx->vkCtx.vkScratch, vkResetFences, ctx->vkCtx.logicalDevice, 1, &fence);
    VK_CALL(ctx->vkCtx.vkScratch, vkQueueSubmit, ctx->graphicsQueue, 1, &submitInfo2, fence);
    VK_CALL(ctx->vkCtx.vkScratch, vkWaitForFences, ctx->vkCtx.logicalDevice, 1, &fence, true, ~u64(0));

    ReleaseResource(&ctx->fencePool, fence);
    ReleaseResource(&ctx->transferCmdPool, cmd);
    ReleaseResource(&ctx->graphicsCmdPool, ownership);

    return ret;
}


void CreateSwapChainFrames(RenderContext* ctx) {

    u32 imageCount = 0;
    VK_CALL(ctx->vkCtx.vkScratch, vkGetSwapchainImagesKHR, ctx->vkCtx.logicalDevice, ctx->swapChain, &imageCount, nullptr);  
    VkImage images[imageCount];
    VK_CALL(ctx->vkCtx.vkScratch, vkGetSwapchainImagesKHR, ctx->vkCtx.logicalDevice, ctx->swapChain, &imageCount, images);
    ctx->swapChainFrames.SetCapacity(&ctx->pointerPool, imageCount);
    ctx->swapChainFrames.size = imageCount;

    auto depthFormat = FindDepthFormat(&ctx->vkCtx);

    for(u32 i = 0; i < ctx->swapChainFrames.size; i++) {

        ctx->swapChainFrames[i].colorImg = images[i];
        ctx->swapChainFrames[i].depthImg.img = CreateImg2D(ctx, {ctx->width, ctx->height}, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        ctx->swapChainFrames[i].depthImg.memory = BackImgMemory(&ctx->vkCtx, ctx->deviceMemory, &ctx->gpuAllocator, ctx->swapChainFrames[i].depthImg.img);

        VkImageViewCreateInfo depthCreateInfo{};
        depthCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthCreateInfo.image    = ctx->swapChainFrames[i].depthImg.img;
        depthCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthCreateInfo.format   = depthFormat;

        depthCreateInfo.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthCreateInfo.subresourceRange.baseMipLevel    = 0;
        depthCreateInfo.subresourceRange.levelCount      = 1;
        depthCreateInfo.subresourceRange.baseArrayLayer  = 0;
        depthCreateInfo.subresourceRange.layerCount      = 1;

        VK_CALL(ctx->vkCtx.vkScratch, vkCreateImageView, ctx->vkCtx.logicalDevice, &depthCreateInfo, &ctx->vkCtx.vkAllocator, &ctx->swapChainFrames[i].depthImgView);

        VkImageViewCreateInfo colorCreateInfo{};
        colorCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorCreateInfo.image    = ctx->swapChainFrames[i].colorImg;
        colorCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorCreateInfo.format   = ctx->swapChainImageFormat;

        colorCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        colorCreateInfo.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
        colorCreateInfo.subresourceRange.baseMipLevel    = 0;
        colorCreateInfo.subresourceRange.levelCount      = 1;
        colorCreateInfo.subresourceRange.baseArrayLayer  = 0;
        colorCreateInfo.subresourceRange.layerCount      = 1;

        VK_CALL(ctx->vkCtx.vkScratch, vkCreateImageView, ctx->vkCtx.logicalDevice, &colorCreateInfo, &ctx->vkCtx.vkAllocator, &ctx->swapChainFrames[i].colorImgView);

    }

}

VkShaderModule CreateShaderModule(VkContext* ctx, const char* source, u32 len) {

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = len;
    createInfo.pCode = (u32*)source;

    VkShaderModule shaderModule;
    VK_CALL(ctx->vkScratch, vkCreateShaderModule, ctx->logicalDevice, &createInfo, &ctx->vkAllocator, &shaderModule);

    return shaderModule;
}

VkDescriptorSetLayout CreateDescriptorSetLayouts(VkContext* ctx, VkDevice logicalDevice, VkDescriptorSetLayoutBinding* bindings, VkDescriptorBindingFlags* flags, u32 count) {

    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extendedInfo{};
    extendedInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
    extendedInfo.bindingCount = count;
    extendedInfo.pBindingFlags = flags;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = count;
    layoutInfo.pBindings = bindings;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;
    layoutInfo.pNext = &extendedInfo;

    VkDescriptorSetLayout ret;
    VK_CALL(ctx->vkScratch, vkCreateDescriptorSetLayout, logicalDevice, &layoutInfo, &ctx->vkAllocator, &ret);
    return ret;
}

VkPipelineLayout CreatePipelineLayout(VkContext* ctx, VkDescriptorSetLayout* descriptorSetLayout, u32 count) {

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = count;
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayout;

    VkPipelineLayout ret;
    VK_CALL(ctx->vkScratch, vkCreatePipelineLayout, ctx->logicalDevice, &pipelineLayoutInfo, &ctx->vkAllocator, &ret);

    return ret;
}
VkRenderPass CreateRenderPass(VkContext* ctx, VkFormat swapChainFormat) {

    auto depthFormat = FindDepthFormat(ctx);

    VkAttachmentDescription attachment[2]{};
    attachment[0].format = swapChainFormat;
    attachment[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachment[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    attachment[1].format = depthFormat;
    attachment[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachment[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;

    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = attachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkRenderPass renderPass;
    VK_CALL(ctx->vkScratch, vkCreateRenderPass, ctx->logicalDevice, &renderPassInfo, &ctx->vkAllocator, &renderPass);

    return renderPass;
}

void CreateGraphicsPipeline(RenderContext* ctx) {

    auto allocSave = ctx->vkCtx.vkScratch;
    auto vertSource = (char*)linear_allocator_top(&ctx->vkCtx.vkScratch);
    u64 vertSourceSize = ReadFile("./vertex.spv", (byte*)vertSource);
    if(vertSourceSize == ~u64(0)) {
        VkLog(&ctx->ioLogBuffer, "s", "File not found");
    }
    linear_allocate(&ctx->vkCtx.vkScratch, vertSourceSize);

    auto fragSource = (char*)linear_allocator_top(&ctx->vkCtx.vkScratch);
    u64 fragSourceSize = ReadFile("./frag.spv", (byte*)fragSource);
    if(fragSourceSize == ~u64(0)) {
        VkLog(&ctx->ioLogBuffer, "s", "File not found");
    }
    linear_allocate(&ctx->vkCtx.vkScratch, fragSourceSize);

    auto vertexModule = CreateShaderModule(&ctx->vkCtx, vertSource, vertSourceSize);
    auto fragmentModule = CreateShaderModule(&ctx->vkCtx, fragSource, fragSourceSize);

    VkPipelineShaderStageCreateInfo shaderStages[2];
    shaderStages[0] = {};
    shaderStages[0].sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage   = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module  = vertexModule;
    shaderStages[0].pName   = "main";

    shaderStages[1] = {};
    shaderStages[1].sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage   = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module  = fragmentModule;
    shaderStages[1].pName   = "main";

    VkVertexInputBindingDescription bindingDescription[2]{};
    bindingDescription[0].binding = 0;
    bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescription[0].stride = sizeof(Vertex);

    bindingDescription[1].binding = 1;
    bindingDescription[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    bindingDescription[1].stride = sizeof(InstanceInfo);

    VkVertexInputAttributeDescription attributeDescription[7]{};
    attributeDescription[0].binding = 0;
    attributeDescription[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescription[0].location = 0;
    attributeDescription[0].offset = offsetof(Vertex, pos);

    attributeDescription[1].binding = 0;
    attributeDescription[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescription[1].location = 1;
    attributeDescription[1].offset = offsetof(Vertex, uv);

    for(u32 i = 0; i < 4; i++) {
        attributeDescription[i + 2].binding = 1;
        attributeDescription[i + 2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescription[i + 2].location = i + 2;
        attributeDescription[i + 2].offset = i * sizeof(vec<f32,3>);
    }
    attributeDescription[6].binding = 1;
    attributeDescription[6].format = VK_FORMAT_R32_UINT;
    attributeDescription[6].location = 6;
    attributeDescription[6].offset = offsetof(InstanceInfo, textureIndex);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 2;
    vertexInputInfo.pVertexBindingDescriptions = bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 7;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkDynamicState dynamicState[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = 2;
    dynamicStateInfo.pDynamicStates = dynamicState;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (f32)ctx->width;
    viewport.height = (f32)ctx->height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent.width = ctx->width;
    scissor.extent.height = ctx->height;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_FALSE;

    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo{};
    depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateInfo.depthTestEnable = VK_TRUE;
    depthStencilStateInfo.depthWriteEnable = VK_TRUE;
    depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilStateInfo.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateInfo.minDepthBounds = 0.0f;
    depthStencilStateInfo.maxDepthBounds = 1.0f;
    depthStencilStateInfo.stencilTestEnable = VK_FALSE;
    depthStencilStateInfo.front = {};
    depthStencilStateInfo.back = {};

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = ctx->pipelineLayout;
    pipelineInfo.renderPass = ctx->renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.pDepthStencilState = &depthStencilStateInfo;

    VK_CALL(ctx->vkCtx.vkScratch, vkCreateGraphicsPipelines, ctx->vkCtx.logicalDevice, nullptr, 1, &pipelineInfo, &ctx->vkCtx.vkAllocator, &ctx->graphicsPipeline);

    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyShaderModule, ctx->vkCtx.logicalDevice, vertexModule, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyShaderModule, ctx->vkCtx.logicalDevice, fragmentModule, &ctx->vkCtx.vkAllocator);

    ctx->vkCtx.vkScratch = allocSave;
}

void CreateFramebuffers(RenderContext* ctx) {
    
    for (size_t i = 0; i < ctx->swapChainFrames.size; i++) {

        VkImageView attachments[] = {ctx->swapChainFrames[i].colorImgView, ctx->swapChainFrames[i].depthImgView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = ctx->renderPass;
        framebufferInfo.attachmentCount = SIZE_OF_ARRAY(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = ctx->width;
        framebufferInfo.height = ctx->height;
        framebufferInfo.layers = 1;

        VK_CALL(ctx->vkCtx.vkScratch , vkCreateFramebuffer, ctx->vkCtx.logicalDevice, &framebufferInfo, &ctx->vkCtx.vkAllocator, &ctx->swapChainFrames[i].frameBuffer);
    }
}
void CreateCommandPool(RenderContext* ctx) {

    VkCommandPoolCreateInfo graphicsPoolInfo{};
    graphicsPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    graphicsPoolInfo.queueFamilyIndex = ctx->families.graphicsFamily;
    graphicsPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPoolCreateInfo transferPoolInfo{};
    transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transferPoolInfo.queueFamilyIndex = ctx->families.transferFamily;
    transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CALL(ctx->vkCtx.vkScratch, vkCreateCommandPool, ctx->vkCtx.logicalDevice, &graphicsPoolInfo, &ctx->vkCtx.vkAllocator, &ctx->commandPoolGraphics);
    VK_CALL(ctx->vkCtx.vkScratch, vkCreateCommandPool, ctx->vkCtx.logicalDevice, &transferPoolInfo, &ctx->vkCtx.vkAllocator, &ctx->commandPoolTransfer);
}
void CreateCommandBuffers(RenderContext* ctx, u32 graphicsCmdCount, u32 transferCmdCount) {
    
    VkCommandBufferAllocateInfo graphicsAllocInfo{};
    graphicsAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    graphicsAllocInfo.commandPool = ctx->commandPoolGraphics;
    graphicsAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    graphicsAllocInfo.commandBufferCount = graphicsCmdCount;

    ctx->graphicsCmdPool.resourceCount = graphicsCmdCount;
    ctx->graphicsCmdPool.resources = (VkCommandBuffer*)local_malloc(&ctx->vkCtx.vkHeap, sizeof(VkCommandBuffer) * graphicsCmdCount);
    VK_CALL(ctx->vkCtx.vkScratch, vkAllocateCommandBuffers, ctx->vkCtx.logicalDevice, &graphicsAllocInfo, ctx->graphicsCmdPool.resources);

    VkCommandBufferAllocateInfo transferAllocInfo{};
    transferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    transferAllocInfo.commandPool = ctx->commandPoolTransfer;
    transferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    transferAllocInfo.commandBufferCount = transferCmdCount;

    ctx->transferCmdPool.resourceCount = transferCmdCount;
    ctx->transferCmdPool.resources = (VkCommandBuffer*)local_malloc(&ctx->vkCtx.vkHeap, sizeof(VkCommandBuffer) * transferCmdCount);
    VK_CALL(ctx->vkCtx.vkScratch, vkAllocateCommandBuffers, ctx->vkCtx.logicalDevice, &transferAllocInfo, ctx->transferCmdPool.resources);
}
const char* VK_MEMORY_PROPERTY_STR[] = {
    "VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT",
    "VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT",
    "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT",
    "VK_MEMORY_PROPERTY_HOST_CACHED_BIT",
    "VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT",
    "VK_MEMORY_PROPERTY_PROTECTED_BIT",
    "VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD",
    "VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD",
    "VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV",
};

void PrintMemoryPropertyFlags(VkMemoryPropertyFlags bits) {

    for(u32 i = 0; i < 9; i++) {
        if( (bits & (1 << i)) == (1 << i)) {
            global_print("sc", VK_MEMORY_PROPERTY_STR[i], '\n');
        }
    }
}
void PrintMemoryTypeBitsFlags(VkContext* ctx, u32 typeFilter) {

    VkPhysicalDeviceMemoryProperties memProperties;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceMemoryProperties, ctx->device, &memProperties);

    for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {
        if( (typeFilter & (1 << i)) == (1 << i) ) {
            global_print("uc", i, '\n');
            PrintMemoryPropertyFlags(memProperties.memoryTypes[i].propertyFlags);
        }
    }
}
u32 MatchMemoryType(VkContext* ctx, u32 typeFilter, VkMemoryPropertyFlags properties) {
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(ctx->device, &memProperties);

    for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {
        if( ((typeFilter & (1 << i)) == (1 << i)) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
            return i;
        }
    }

    return ~u32(0);
}

u32 GetImportMemoryAlignment(VkContext* ctx, VkPhysicalDevice device) {

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT al{};
    al.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 st{};
    st.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    st.pNext = &al;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceProperties2, device, &st);
    return al.minImportedHostPointerAlignment;
}

VkDeviceMemory ImportMemory(VkContext* ctx, void* mem, u32 size, u32 typeIndex) {

    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkImportMemoryHostPointerInfoEXT import{};
    import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import.pHostPointer = mem;
    import.pNext = &flags;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = typeIndex;
    allocInfo.pNext = &import;
    
    VkDeviceMemory memory = nullptr;
    VK_CALL(ctx->vkScratch, vkAllocateMemory, ctx->logicalDevice, &allocInfo, &ctx->vkAllocator, &memory);

    return memory;
}

VkDeviceMemory AllocateGPUMemory(VkContext* ctx, u32 size, u32 typeIndex) {

    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = typeIndex;
    allocInfo.pNext = &flags;

    VkDeviceMemory memory = nullptr;
    VK_CALL(ctx->vkScratch, vkAllocateMemory, ctx->logicalDevice, &allocInfo, &ctx->vkAllocator, &memory);
    return memory;
}

struct VkBufferArgs {
    u32* queueFamilyIndicies;
    u32 queueFamilyCount;
    VkBufferUsageFlags usage;
    VkSharingMode sharing;
    bool externaMem;
};
VkBuffer MakeVkBuffer(VkContext* ctx, u32 size, VkBufferArgs args) {

    VkExternalMemoryBufferCreateInfo ext;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = args.usage;
    bufferInfo.sharingMode = args.sharing;
    bufferInfo.queueFamilyIndexCount = args.queueFamilyCount;
    bufferInfo.pQueueFamilyIndices = args.queueFamilyIndicies;
    
    if(args.externaMem) {
        bufferInfo.pNext = &ext;
        ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        ext.pNext = 0;
    }

    VkBuffer buffer;
    VK_CALL(ctx->vkScratch, vkCreateBuffer, ctx->logicalDevice, &bufferInfo, &ctx->vkAllocator, &buffer);
    return buffer;
}

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

ModelDesciption UploadModel(RenderContext* ctx, LoadedInfo model) {

    ModelDesciption ret;
    ret.indexCount = model.indexSize / sizeof(u32);
    ret.vertexCount = model.vertexSize / sizeof(Vertex);

    VkCommandBuffer transferCmd;
    ASSERT(TryAcquireResource(&ctx->transferCmdPool, &transferCmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CALL(ctx->vkCtx.vkScratch, vkBeginCommandBuffer, transferCmd, &begin);

    auto block = allocate_gpu_block(&ctx->gpuAllocator, model.indexSize + model.vertexSize, 16);
    VkBufferCopy bufferInfo{};
    bufferInfo.size = block.size;
    bufferInfo.srcOffset = model.vertexOffset;
    bufferInfo.dstOffset = block.offset;

    ret.vertexOffset = block.offset;
    ret.indexOffset = block.offset + model.vertexSize;

    VK_CALL(ctx->vkCtx.vkScratch, vkCmdCopyBuffer, transferCmd, ctx->hostBuffer, ctx->deviceBuffer, 1, &bufferInfo);
    VK_CALL(ctx->vkCtx.vkScratch, vkEndCommandBuffer, transferCmd);

    VkFence finished;
    ASSERT(TryAcquireResource(&ctx->fencePool, &finished));
    VK_CALL(ctx->vkCtx.vkScratch, vkResetFences, ctx->vkCtx.logicalDevice, 1, &finished);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transferCmd;
    
    VK_CALL(ctx->vkCtx.vkScratch, vkQueueSubmit, ctx->transferQueue, 1, &submit, finished);
    VK_CALL(ctx->vkCtx.vkScratch, vkWaitForFences, ctx->vkCtx.logicalDevice, 1, &finished, 1, ~u64(0));

    ReleaseResource(&ctx->fencePool, finished);
    ReleaseResource(&ctx->transferCmdPool, transferCmd);

    ctx->uploadMemory.top = model.vertexOffset;

    return ret;
}
void RecreateSwapChain(RenderContext* ctx, u32 width, u32 height) {

    ctx->width = width;
    ctx->height = height;

    VK_CALL(ctx->vkCtx.vkScratch, vkDeviceWaitIdle, ctx->vkCtx.logicalDevice);
    for(u32 i = 0; i < ctx->swapChainFrames.size; i++) {
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyImageView, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].colorImgView, &ctx->vkCtx.vkAllocator);
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyImageView, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].depthImgView, &ctx->vkCtx.vkAllocator);
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyImage, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].depthImg.img, &ctx->vkCtx.vkAllocator);
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyFramebuffer, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].frameBuffer, &ctx->vkCtx.vkAllocator);
        free_gpu_block(&ctx->gpuAllocator, ctx->swapChainFrames[i].depthImg.memory);
    }
    
    auto newChain = CreateSwapChain(&ctx->vkCtx, {width, height}, ctx->vkCtx.device, ctx->vkCtx.logicalDevice, ctx->surface, ctx->families, ctx->swapChain);
    ctx->swapChain = newChain.swapChain;
    ctx->swapChainImageFormat = newChain.format;

    CreateSwapChainFrames(ctx);
    CreateFramebuffers(ctx);

    VkLog(&ctx->ioLogBuffer, "susus", "[vulkan info] swapchain recreated(", ctx->width, ", ", ctx->height, ")\n");
    VK_CALL(ctx->vkCtx.vkScratch, vkDeviceWaitIdle, ctx->vkCtx.logicalDevice);
}

VkFence CreateFence(VkContext* ctx) {
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence ret;
    VK_CALL(ctx->vkScratch, vkCreateFence, ctx->logicalDevice, &fenceInfo, &ctx->vkAllocator, &ret);
    return ret;
}
VkSemaphore CreateSemaphore(VkContext* ctx) {
    
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore ret;
    VK_CALL(ctx->vkScratch, vkCreateSemaphore, ctx->logicalDevice, &semaphoreInfo, &ctx->vkAllocator, &ret);
    return ret;
}
void CreateSyncObjects(RenderContext* ctx, u32 count) {

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    ctx->fencePool.resourceCount = count;
    ctx->fencePool.resources = (VkFence*)local_malloc(&ctx->vkCtx.vkHeap, sizeof(VkFence) * ctx->fencePool.resourceCount);

    for(u32 i = 0; i < ctx->fencePool.resourceCount; i++) {
        VK_CALL(ctx->vkCtx.vkScratch, vkCreateFence, ctx->vkCtx.logicalDevice, &fenceInfo, &ctx->vkCtx.vkAllocator, ctx->fencePool.resources + i);
    }

    ctx->semaphorePool.resourceCount = count;
    ctx->semaphorePool.resources = (VkSemaphore*)local_malloc(&ctx->vkCtx.vkHeap , sizeof(VkSemaphore) * ctx->semaphorePool.resourceCount);
    for(u32 i = 0; i < ctx->semaphorePool.resourceCount; i++) {
        VK_CALL(ctx->vkCtx.vkScratch, vkCreateSemaphore, ctx->vkCtx.logicalDevice, &semaphoreInfo, &ctx->vkCtx.vkAllocator, ctx->semaphorePool.resources + i);
    }
}

struct VkHeaps {
    CoalescingLinearAllocator uploadAllocator;
    GpuHeap deviceAllocator;

    VkDeviceMemory deviceMemory;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkBuffer hostBuffer;


    u64 deviceHeapGpuAddress;
    u64 hostHeapGpuAddress;
};

VkHeaps CreateHeaps(RenderContext* ctx, byte* localMemory, u32 localMemorySize, u32 gpuAllocatorSize) {

    VkHeaps heap{};

    auto aligment = GetImportMemoryAlignment(&ctx->vkCtx, ctx->vkCtx.device);
    auto aligned = (byte*)align_pointer(localMemory, aligment);
    localMemorySize -= aligned - localMemory;

    heap.uploadAllocator = make_coalescing_linear_allocator(aligned, localMemorySize);

    auto vkGetMemoryHostPointerPropertiesEXT_ = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(ctx->vkCtx.logicalDevice, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT prop{};
    prop.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    VK_CALL(ctx->vkCtx.vkScratch, vkGetMemoryHostPointerPropertiesEXT_, ctx->vkCtx.logicalDevice, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, heap.uploadAllocator.base, &prop);
    
    auto importMemoryType = MatchMemoryType(&ctx->vkCtx, prop.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    heap.hostMemory = ImportMemory(&ctx->vkCtx, heap.uploadAllocator.base, heap.uploadAllocator.cap, importMemoryType);

    auto deviceMemoryType = MatchMemoryType(&ctx->vkCtx, ~u32(0), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    heap.deviceMemory = AllocateGPUMemory(&ctx->vkCtx, gpuAllocatorSize, deviceMemoryType);

    ctx->gpuAllocator.used_blocks[0].size = gpuAllocatorSize;
    ctx->gpuAllocator.used_block_count = 1;

    u32 uniqe[4];
    memcpy(uniqe, &ctx->families, sizeof(u32) * 4);
    u32 uniqeCount = get_unique(uniqe, 4);

    auto usageHost = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT   |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferArgs argsHost{uniqe, uniqeCount, usageHost, VK_SHARING_MODE_CONCURRENT, true};
    heap.hostBuffer = MakeVkBuffer(&ctx->vkCtx, heap.uploadAllocator.cap, argsHost);

    auto usageDecie = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT   |
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferArgs argsDev{uniqe, uniqeCount, usageDecie, VK_SHARING_MODE_CONCURRENT, false};
    heap.deviceBuffer = MakeVkBuffer(&ctx->vkCtx, gpuAllocatorSize, argsDev);

    VK_CALL(ctx->vkCtx.vkScratch, vkBindBufferMemory, ctx->vkCtx.logicalDevice, heap.hostBuffer, heap.hostMemory, 0);
    VK_CALL(ctx->vkCtx.vkScratch, vkBindBufferMemory, ctx->vkCtx.logicalDevice, heap.deviceBuffer, heap.deviceMemory, 0);

    auto VkGetAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(ctx->vkCtx.logicalDevice, "vkGetBufferDeviceAddressKHR");
    VkBufferDeviceAddressInfo hostAdddessInfo{};
    hostAdddessInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    hostAdddessInfo.buffer = heap.hostBuffer;
    VkBufferDeviceAddressInfo deviceAdddessInfo{};
    deviceAdddessInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    deviceAdddessInfo.buffer = heap.deviceBuffer;

    heap.hostHeapGpuAddress = VkGetAddressKHR(ctx->vkCtx.logicalDevice, &hostAdddessInfo);;
    heap.deviceHeapGpuAddress = VkGetAddressKHR(ctx->vkCtx.logicalDevice, &deviceAdddessInfo);;

    return heap;
}

void T() {

    static const u32 k_max_bindless_resources = 16536;
    VkDescriptorPoolSize pool_sizes_bindless[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k_max_bindless_resources }
    };
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;
    pool_info.maxSets = k_max_bindless_resources * SIZE_OF_ARRAY( pool_sizes_bindless );
    pool_info.poolSizeCount = (u32)SIZE_OF_ARRAY(pool_sizes_bindless);
    pool_info.pPoolSizes = pool_sizes_bindless;

    VkDescriptorPool vulkan_descriptor_pool_bindless;
    auto result = vkCreateDescriptorPool( 0, &pool_info, 0, &vulkan_descriptor_pool_bindless);

    VkDescriptorBindingFlags bindless_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;

    VkDescriptorSetLayoutBinding vk_binding;
    vk_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vk_binding.descriptorCount = k_max_bindless_resources;
    vk_binding.binding = 0;
    vk_binding.stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extended_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT, nullptr };
    extended_info.bindingCount = 1;
    extended_info.pBindingFlags = &bindless_flags;

    VkDescriptorSetLayoutCreateInfo layout_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout_info.bindingCount = 1;
    layout_info.pBindings = &vk_binding;
    layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;
    layout_info.pNext = &extended_info;

    VkDescriptorSetLayout vulkan_bindless_descriptor_layout;
    vkCreateDescriptorSetLayout(0, &layout_info, 0, &vulkan_bindless_descriptor_layout);

    VkDescriptorSetVariableDescriptorCountAllocateInfoEXT count_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT };
    u32 max_binding = k_max_bindless_resources - 1;
    count_info.descriptorSetCount = 1;
    count_info.pDescriptorCounts = &max_binding;

    VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    alloc_info.descriptorPool = vulkan_descriptor_pool_bindless;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &vulkan_bindless_descriptor_layout;
    alloc_info.pNext = &count_info;

    VkDescriptorSet vulkan_bindless_descriptor_set;
    vkAllocateDescriptorSets(0, &alloc_info, &vulkan_bindless_descriptor_set);
}


VkDescriptorPool CreateDescriptorPool(VkContext* ctx, u32 setCount, u32 uboCount, u32 samplerCount) {

    VkDescriptorPoolSize poolSize[2]{};
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = uboCount;
    poolSize[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize[1].descriptorCount = samplerCount;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSize;
    poolInfo.maxSets = setCount;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;

    VkDescriptorPool ret;
    VK_CALL(ctx->vkScratch, vkCreateDescriptorPool, ctx->logicalDevice, &poolInfo, &ctx->vkAllocator, &ret);
    return ret;
}

struct DescriptorSetsAllocateInfo {
    VkDescriptorSetLayout layout;
    VkDescriptorSet* result;
    u32 count;
};
void AllocateDescriptorSets(VkContext* ctx, VkDescriptorPool pool, u32 count, VkDescriptorSetLayout layout, u32 variableDescriptorCount, VkDescriptorSet* result) {
    
    VkDescriptorSetLayout layouts[count];
    for(u32 i = 0; i < count; i++) {
        layouts[i] = layout;
    }

    u32 variableDescriptor[count];
    for(u32 i = 0; i < count; i++) {
        variableDescriptor[i] = variableDescriptorCount;
    }
    VkDescriptorSetVariableDescriptorCountAllocateInfoEXT countInfo{};
    countInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
    countInfo.descriptorSetCount = count;
    countInfo.pDescriptorCounts = variableDescriptor;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = count;
    allocInfo.pSetLayouts = layouts;
    allocInfo.pNext = &countInfo;

    VK_CALL(ctx->vkScratch, vkAllocateDescriptorSets, ctx->logicalDevice, &allocInfo, result);
}

void GetDeviceName(VkContext* ctx, VkPhysicalDevice device, char* name) {

    VkPhysicalDeviceProperties deviceProperties{};
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceProperties, device, &deviceProperties);
    str_cpy(name, deviceProperties.deviceName);
}
void CreatePipeline(RenderContext* ctx) {

    auto save = ctx->vkCtx.vkScratch;
    auto vertSource = (char*)linear_allocator_top(&ctx->vkCtx.vkScratch);
    u64 vertSourceSize = ReadFile("./vertex.spv", (byte*)vertSource);
    if(vertSourceSize == ~u64(0)) {
        VkLog(&ctx->ioLogBuffer, "s", "File not found");
    }
    linear_allocate(&ctx->vkCtx.vkScratch, vertSourceSize);

    auto fragSource = (char*)linear_allocator_top(&ctx->vkCtx.vkScratch);
    u64 fragSourceSize = ReadFile("./frag.spv", (byte*)fragSource);
    if(fragSourceSize == ~u64(0)) {
        VkLog(&ctx->ioLogBuffer, "s", "File not found");
    }
    linear_allocate(&ctx->vkCtx.vkScratch, fragSourceSize);

    auto vertexModule = CreateShaderModule(&ctx->vkCtx, vertSource, vertSourceSize);
    auto fragmentModule = CreateShaderModule(&ctx->vkCtx, fragSource, fragSourceSize);

    VkPipelineShaderStageCreateInfo shaderStages[2];
    shaderStages[0] = {};
    shaderStages[0].sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage   = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module  = vertexModule;
    shaderStages[0].pName   = "main";

    shaderStages[1] = {};
    shaderStages[1].sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage   = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module  = fragmentModule;
    shaderStages[1].pName   = "main";

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindingDescription.stride = sizeof(Vertex);

    VkVertexInputAttributeDescription attributeDescription[2]{};
    attributeDescription[0].binding = 0;
    attributeDescription[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescription[0].location = 0;
    attributeDescription[0].offset = offsetof(Vertex, pos);

    attributeDescription[1].binding = 0;
    attributeDescription[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescription[1].location = 1;
    attributeDescription[1].offset = offsetof(Vertex, uv);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (f32)ctx->width;
    viewport.height = (f32)ctx->height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent.width = ctx->width;
    scissor.extent.height = ctx->height;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_FALSE;

    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo{};
    depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateInfo.depthTestEnable = VK_TRUE;
    depthStencilStateInfo.depthWriteEnable = VK_TRUE;
    depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilStateInfo.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateInfo.minDepthBounds = 0.0f;
    depthStencilStateInfo.maxDepthBounds = 1.0f;
    depthStencilStateInfo.stencilTestEnable = VK_FALSE;
    depthStencilStateInfo.front = {};
    depthStencilStateInfo.back = {};

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = ctx->pipelineLayout;
    pipelineInfo.renderPass = ctx->renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.pDepthStencilState = &depthStencilStateInfo;

    VK_CALL(ctx->vkCtx.vkScratch, vkCreateGraphicsPipelines, ctx->vkCtx.logicalDevice, nullptr, 1, &pipelineInfo, &ctx->vkCtx.vkAllocator, &ctx->graphicsPipeline);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyShaderModule, ctx->vkCtx.logicalDevice, vertexModule, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyShaderModule, ctx->vkCtx.logicalDevice, fragmentModule, &ctx->vkCtx.vkAllocator);

    ctx->vkCtx.vkScratch = save;
}

void AddDescriptorUpdate(PendingDescriptorUpdates* pending, VkWriteDescriptorSet write, void* info) {
    auto i = pending->count++;
    pending->write[i] = write;
    memcpy(pending->infos + i, info, sizeof(VkDescriptorBufferInfo));
}

void UnRegisterTexture(RenderContext* ctx, u32 slot) {

    ctx->textureSlotTable[slot] = ctx->head;
    ctx->head = slot;
}
u32 RegisterTexture(RenderContext* ctx, VkTextureInfo texture, VkSampler sampler, VkDescriptorSet set) {

    auto slot = ctx->head;
    ctx->head = ctx->textureSlotTable[slot];
    
    auto updateI = ctx->descriptorUpdates.count++;
    auto textureDescriptorWrite = &ctx->descriptorUpdates.write[updateI];
    auto imgInfo = &ctx->descriptorUpdates.infos[updateI].imgInfo;

    imgInfo->sampler = sampler;
    imgInfo->imageView = texture.view;
    imgInfo->imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    *textureDescriptorWrite = {};
    textureDescriptorWrite->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    textureDescriptorWrite->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureDescriptorWrite->descriptorCount = 1;
    textureDescriptorWrite->dstBinding = 0;
    textureDescriptorWrite->dstSet = set;
    textureDescriptorWrite->dstArrayElement = slot;
    textureDescriptorWrite->pImageInfo = imgInfo;
    
    return slot;
}

void MakeRenderContext(RenderContext* ctx, byte* memory, u32 memorySize) {

    if(!glfwVulkanSupported()) {
        VkLog(&ctx->ioLogBuffer, "s", "Vulkan not supported\n");
        ASSERT(false);
    }

    memset(memory, 0, memorySize);
    *ctx = {};
    ctx->ioLogBuffer = make_linear_allocator(memory, 1024*2);
    memory += 1024*2;
    memorySize -= 1024*2;

    ctx->pointerPool = make_multi_memory_pool<sizeof(void*)>(memory, Kilobyte(4));
    memory += Kilobyte(4);
    memorySize -= Kilobyte(4);

    ctx->vkCtx.vkScratch = make_linear_allocator(memory, Megabyte(4));
    memory += Megabyte(4);
    memorySize -= Megabyte(4);

    ctx->gpuAllocator = make_gpu_heap(memory, 256, 0);
    memory += 256 * sizeof(GpuMemoryBlock);
    memorySize -= 256 * sizeof(GpuMemoryBlock);

    ctx->vkCtx.vkHeap = make_local_malloc(memory, Megabyte(32));
    memory += Megabyte(32);
    memorySize -= Megabyte(32);

    ctx->head = 0;
    ctx->textureSlotTable[511] = ~u16(0);
    for(u32 i = 0; i < 511; i++) {
        ctx->textureSlotTable[i] = i + 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    ctx->width = 300;
    ctx->height = 300;
    ctx->title = "vk_test";
    GLFW_CALL(ctx->glfwHandle = glfwCreateWindow(ctx->width, ctx->height, ctx->title, nullptr, nullptr));
    GLFW_CALL(glfwSetWindowUserPointer(ctx->glfwHandle, ctx));

    auto icon = LoadBMP("../res/Spaceship.bmp", linear_allocator_top(&ctx->vkCtx.vkScratch));
    GLFW_CALL(glfwSetWindowIcon(ctx->glfwHandle, 1, (GLFWimage*)&icon));
    VkLog(&ctx->ioLogBuffer, "sssicic", "[glfw info] window \"", ctx->title, "\" created ", ctx->width, ' ', ctx->height, '\n');

    ctx->vkCtx.vkAllocator = {};
    ctx->vkCtx.vkAllocator.pUserData       = &ctx->vkCtx;
    ctx->vkCtx.vkAllocator.pfnAllocation   = vkLocalMalloc;
    ctx->vkCtx.vkAllocator.pfnFree         = vkLocalFree;
    ctx->vkCtx.vkAllocator.pfnReallocation = vkLocalRealloc;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = ctx->title;
    appInfo.pEngineName = ctx->title;
    appInfo.applicationVersion  = VK_MAKE_VERSION(1, 0, 0);
    appInfo.engineVersion       = VK_MAKE_VERSION(1, 0, 0);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkFlushLog(&ctx->ioLogBuffer);
    VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo{};
    if constexpr(ENABLE_VALIDATION_LAYER) {

        LOG_ASSERT(CheckValidationLayerSupport(validationLayers, SIZE_OF_ARRAY(validationLayers)), "validation layers not found");
        VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] validation layers supported\n");

        createInfo.enabledLayerCount = SIZE_OF_ARRAY(validationLayers);
        createInfo.ppEnabledLayerNames = validationLayers;

        ctx->logSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
        VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] debug messenger severity set to: ERROR WARNING\n");

        debugMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugMessengerCreateInfo.messageSeverity =  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        debugMessengerCreateInfo.messageType =  VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugMessengerCreateInfo.pfnUserCallback = DebugCallback;
        debugMessengerCreateInfo.pUserData = ctx;
        createInfo.pNext = &debugMessengerCreateInfo;
    }

    auto requiredExt = (const char**)linear_allocator_top(&ctx->vkCtx.vkScratch);
    createInfo.enabledExtensionCount = GetRequiredExtensions(requiredExt, instanceExtensions, SIZE_OF_ARRAY(instanceExtensions));
    createInfo.ppEnabledExtensionNames = requiredExt;
    ctx->vkCtx.vkScratch.top += sizeof(const char**) * createInfo.enabledExtensionCount;
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] required instance extentions found\n");

    VK_CALL(ctx->vkCtx.vkScratch, vkCreateInstance, &createInfo, &ctx->vkCtx.vkAllocator, &ctx->vkCtx.vkInstance);
    ctx->vkCtx.vkScratch.top -= sizeof(const char**) * createInfo.enabledExtensionCount;
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] instance created\n");

    if constexpr (ENABLE_VALIDATION_LAYER) {
        ctx->vkCtx.vkDebugMessenger = CreateDebugUtilsMessengerEXT(&ctx->vkCtx, &debugMessengerCreateInfo);
        VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] debug messenger created\n");
    }
    GLFW_CALL(glfwCreateWindowSurface(ctx->vkCtx.vkInstance, ctx->glfwHandle, &ctx->vkCtx.vkAllocator, &ctx->surface));

    ctx->vkCtx.device = PickPhysicalDevice(&ctx->vkCtx, ctx->vkCtx.vkInstance, ctx->surface, deviceExtensions, SIZE_OF_ARRAY(deviceExtensions));
    if(!ctx->vkCtx.device) {
        VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] no gpu with vulkan support was found\n");
        ASSERT(false);
    }
    ctx->families = GetQueueFamilies(&ctx->vkCtx, ctx->vkCtx.device, ctx->surface);

    char devName[256];
    GetDeviceName(&ctx->vkCtx, ctx->vkCtx.device, devName);
    VkLog(&ctx->ioLogBuffer, "ssc", "[vulkan info] physical device found: ", devName, '\n');

    ctx->vkCtx.logicalDevice = CreateLogicalDevice(&ctx->vkCtx, ctx->families, ctx->vkCtx.device);
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] created logical device\n");
    VK_CALL(ctx->vkCtx.vkScratch, vkGetDeviceQueue, ctx->vkCtx.logicalDevice, ctx->families.graphicsFamily, 0, &ctx->graphicsQueue);
    VK_CALL(ctx->vkCtx.vkScratch, vkGetDeviceQueue, ctx->vkCtx.logicalDevice, ctx->families.presentFamily,  0, &ctx->presentQueue);
    VK_CALL(ctx->vkCtx.vkScratch, vkGetDeviceQueue, ctx->vkCtx.logicalDevice, ctx->families.computeFamily,  0, &ctx->computeQueue);
    VK_CALL(ctx->vkCtx.vkScratch, vkGetDeviceQueue, ctx->vkCtx.logicalDevice, ctx->families.transferFamily, 0, &ctx->transferQueue);

    auto heap = CreateHeaps(ctx, memory, memorySize, Megabyte(64));
    ctx->uploadMemory               = heap.uploadAllocator;
    ctx->deviceBuffer               = heap.deviceBuffer;
    ctx->hostBuffer                 = heap.hostBuffer;
    ctx->hostMemoryDeviceAddress    = heap.hostHeapGpuAddress;
    ctx->hostMemoryDeviceAddress    = heap.deviceHeapGpuAddress;
    ctx->deviceMemory               = heap.deviceMemory;
    ctx->hostMemory                 = heap.hostMemory;
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] created host and device memory heaps\n");

    auto chain = CreateSwapChain(&ctx->vkCtx, {ctx->width, ctx->height}, ctx->vkCtx.device, ctx->vkCtx.logicalDevice, ctx->surface, ctx->families, nullptr);
    ctx->swapChain = chain.swapChain;
    ctx->swapChainImageFormat = chain.format;

    CreateSwapChainFrames(ctx);
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] created swapchain\n");
    CreateSyncObjects(ctx, 9);
    CreateCommandPool(ctx);
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] created command pools\n");
    CreateCommandBuffers(ctx, 2, 2);
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] created graphics command buffers\n");
    ctx->descriptorPool = CreateDescriptorPool(&ctx->vkCtx, 4, 3, 512);

    VkDescriptorSetLayoutBinding bufferDescriptorBinding{};
    bufferDescriptorBinding.binding = 0;
    bufferDescriptorBinding.descriptorCount = 1;
    bufferDescriptorBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bufferDescriptorBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding textureDescriptorBinding{};
    textureDescriptorBinding.binding = 0;
    textureDescriptorBinding.descriptorCount = 512;
    textureDescriptorBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureDescriptorBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorBindingFlags nullFlag = 0;
    VkDescriptorBindingFlags bindlessFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;

    ctx->UBOdescriptorLayout = CreateDescriptorSetLayouts(&ctx->vkCtx, ctx->vkCtx.logicalDevice, &bufferDescriptorBinding, &nullFlag, 1);
    ctx->textureDescriptorLayout = CreateDescriptorSetLayouts(&ctx->vkCtx, ctx->vkCtx.logicalDevice, &textureDescriptorBinding, &bindlessFlags, 1);

    VkDescriptorSet sets[3];
    AllocateDescriptorSets(&ctx->vkCtx, ctx->descriptorPool, 3, ctx->UBOdescriptorLayout, 0, sets);
    AllocateDescriptorSets(&ctx->vkCtx, ctx->descriptorPool, 1, ctx->textureDescriptorLayout, 512, &ctx->textureDescriptors);
    
    VkWriteDescriptorSet descriptorWrites[3];
    ctx->descriptorSetPool.resourceCount = 3;
    ctx->descriptorSetPool.resources = (Descriptor*)local_malloc(&ctx->vkCtx.vkHeap, sizeof(Descriptor) * ctx->descriptorSetPool.resourceCount);
    for(u32 i = 0; i < ctx->descriptorSetPool.resourceCount; i++) {
        auto block = allocate_gpu_block(&ctx->gpuAllocator, sizeof(CommonParams), 256);
        ctx->descriptorSetPool.resources[i].offset = block.offset;
        ctx->descriptorSetPool.resources[i].set = sets[i];

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = ctx->deviceBuffer;
        bufferInfo.offset = block.offset;
        bufferInfo.range = block.size;

        descriptorWrites[i] = {};
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].dstBinding = 0;
        descriptorWrites[i].dstSet = sets[i];
        descriptorWrites[i].pBufferInfo = &bufferInfo;
    }
    VK_CALL(ctx->vkCtx.vkScratch, vkUpdateDescriptorSets, ctx->vkCtx.logicalDevice, 3, descriptorWrites, 0, nullptr);

    VkDescriptorSetLayout descriptorLayouts[2] = {ctx->UBOdescriptorLayout, ctx->textureDescriptorLayout};
    ctx->pipelineLayout = CreatePipelineLayout(&ctx->vkCtx, descriptorLayouts, 2);
    ctx->renderPass = CreateRenderPass(&ctx->vkCtx, ctx->swapChainImageFormat);
    CreateFramebuffers(ctx);
    CreateGraphicsPipeline(ctx);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    VK_CALL(ctx->vkCtx.vkScratch, vkCreateSampler, ctx->vkCtx.logicalDevice, &samplerInfo, &ctx->vkCtx.vkAllocator, &ctx->textureSampler);

    VK_CALL(ctx->vkCtx.vkScratch, vkDeviceWaitIdle, ctx->vkCtx.logicalDevice);
    VkFlushLog(&ctx->ioLogBuffer);
}

void DestroyRenderContext(RenderContext* ctx) {

    VK_CALL(ctx->vkCtx.vkScratch, vkDeviceWaitIdle, ctx->vkCtx.logicalDevice);
    for(u32 i = 0; i < ctx->semaphorePool.resourceCount; i++) {
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroySemaphore, ctx->vkCtx.logicalDevice, ctx->semaphorePool.resources[i], &ctx->vkCtx.vkAllocator);
    }
    for(u32 i = 0; i < ctx->fencePool.resourceCount; i++) {
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyFence, ctx->vkCtx.logicalDevice, ctx->fencePool.resources[i], &ctx->vkCtx.vkAllocator);
    }
    for(u32 i = 0; i < ctx->swapChainFrames.size; i++) {

        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyImageView, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].depthImgView, &ctx->vkCtx.vkAllocator);
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyImageView, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].colorImgView, &ctx->vkCtx.vkAllocator);
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyImage, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].depthImg.img, &ctx->vkCtx.vkAllocator);
        VK_CALL(ctx->vkCtx.vkScratch, vkDestroyFramebuffer, ctx->vkCtx.logicalDevice, ctx->swapChainFrames[i].frameBuffer, &ctx->vkCtx.vkAllocator);
    }
  
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroySampler, ctx->vkCtx.logicalDevice, ctx->textureSampler, &ctx->vkCtx.vkAllocator);

    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyDescriptorSetLayout, ctx->vkCtx.logicalDevice, ctx->UBOdescriptorLayout, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyDescriptorSetLayout, ctx->vkCtx.logicalDevice, ctx->textureDescriptorLayout, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyBuffer, ctx->vkCtx.logicalDevice, ctx->hostBuffer, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyBuffer, ctx->vkCtx.logicalDevice, ctx->deviceBuffer, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkFreeMemory, ctx->vkCtx.logicalDevice, ctx->hostMemory, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkFreeMemory, ctx->vkCtx.logicalDevice, ctx->deviceMemory, &ctx->vkCtx.vkAllocator);
 
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyDescriptorPool, ctx->vkCtx.logicalDevice, ctx->descriptorPool, &ctx->vkCtx.vkAllocator);

    VK_CALL(ctx->vkCtx.vkScratch, vkFreeCommandBuffers, ctx->vkCtx.logicalDevice, ctx->commandPoolGraphics, ctx->graphicsCmdPool.resourceCount, ctx->graphicsCmdPool.resources);
    VK_CALL(ctx->vkCtx.vkScratch, vkFreeCommandBuffers, ctx->vkCtx.logicalDevice, ctx->commandPoolTransfer, ctx->transferCmdPool.resourceCount, ctx->transferCmdPool.resources);
    
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyCommandPool, ctx->vkCtx.logicalDevice, ctx->commandPoolGraphics, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyCommandPool, ctx->vkCtx.logicalDevice, ctx->commandPoolTransfer, &ctx->vkCtx.vkAllocator);

    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyPipeline, ctx->vkCtx.logicalDevice, ctx->graphicsPipeline, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyRenderPass, ctx->vkCtx.logicalDevice, ctx->renderPass, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyPipelineLayout, ctx->vkCtx.logicalDevice, ctx->pipelineLayout, &ctx->vkCtx.vkAllocator);

    VK_CALL(ctx->vkCtx.vkScratch, vkDestroySwapchainKHR, ctx->vkCtx.logicalDevice, ctx->swapChain, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroySurfaceKHR, ctx->vkCtx.vkInstance, ctx->surface, &ctx->vkCtx.vkAllocator);
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyDevice, ctx->vkCtx.logicalDevice, &ctx->vkCtx.vkAllocator);

    if constexpr(ENABLE_VALIDATION_LAYER) {
        DestroyDebugUtilsMessengerEXT(&ctx->vkCtx, ctx->vkCtx.vkDebugMessenger);
    }
    VK_CALL(ctx->vkCtx.vkScratch, vkDestroyInstance, ctx->vkCtx.vkInstance, &ctx->vkCtx.vkAllocator);
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] vk instance destroyed\n");

    GLFW_CALL(glfwDestroyWindow(ctx->glfwHandle));
    VkLog(&ctx->ioLogBuffer, "sss", "[glfw info] window \"", ctx->title, "\" destroyed\n");
    VkFlushLog(&ctx->ioLogBuffer);
}
void VkBeginCmd(VkContext* ctx, VkCommandBuffer buffer) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CALL(ctx->vkScratch, vkBeginCommandBuffer, buffer, &beginInfo);
}
void Update(EngineState *state) {

    state->time += 0.01;
    state->delta++;
    GLFW_CALL(glfwPollEvents());
    GLFW_CALL(auto F11Status = glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_F11));
    if(state->delta > 60 && F11Status == GLFW_PRESS) {

        GLFWmonitor* monitors[] = {state->primary, nullptr};
        u32 dimensions[2][2] = { {1920,1080}, {900,900} };
        GLFW_CALL(glfwSetWindowMonitor(state->ctx->glfwHandle, monitors[state->fullscreen], 0,0, dimensions[state->fullscreen][0], dimensions[state->fullscreen][1], GLFW_DONT_CARE));
        state->fullscreen = !state->fullscreen;
        state->delta = 0;
    }

    u8 keys = 0;
    GLFW_CALL(keys |= KEY_W & (~u32(0) * glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_W)));
    GLFW_CALL(keys |= KEY_S & (~u32(0) * glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_S)));
    GLFW_CALL(keys |= KEY_A & (~u32(0) * glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_A)));
    GLFW_CALL(keys |= KEY_D & (~u32(0) * glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_D)));
    GLFW_CALL(keys |= KEY_LEFT_SHIFT & (~u32(0) * glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_LEFT_SHIFT)));
    GLFW_CALL(keys |= KEY_SPACE & (~u32(0) * glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_SPACE)));
    GLFW_CALL(auto ctrl = glfwGetKey(state->ctx->glfwHandle, GLFW_KEY_LEFT_CONTROL));

    ComputeCameraVelocity(&state->camera, keys, 0.0001);
    state->camera.position = state->camera.position + state->camera.vel;
    state->camera.vel = state->camera.vel * 0.95;

    if(!ctrl) {

        f64 cursorX;
        f64 cursorY;
        GLFW_CALL(glfwGetCursorPos(state->ctx->glfwHandle, &cursorX, &cursorY));
        f32 verticalAngle = ((state->ctx->height * 0.5) - cursorY) / (f32)(state->ctx->height * 0.5f );
        f32 horizontalAngle = ((state->ctx->width * 0.5) - cursorX) / (f32)(state->ctx->width * 0.5f );
        cursorX = (f64)state->ctx->width * 0.5;
        cursorY = (f64)state->ctx->height * 0.5;
        GLFW_CALL(glfwSetCursorPos(state->ctx->glfwHandle, cursorX, cursorY));
        RotateCamera(&state->camera, verticalAngle, -horizontalAngle);
    }
}

struct DrawInfo {
    ModelDesciption model;
    u32 instanceCount;
};
struct CmdState {
    VkCommandBuffer cmd;
    VkFence complete;

    void* resources;
    void (*onRetire)(RenderContext* ctx, void*);
    ResourcePool<VkCommandBuffer>* cmdSource;
};

bool TryRetireCmdState(RenderContext* ctx, CmdState* cmdState) {

    auto save = ctx->vkCtx.vkScratch.top;
    auto status = vkGetFenceStatus(ctx->vkCtx.logicalDevice, cmdState->complete);

    if(status == VK_SUCCESS) {
        ReleaseResource(&ctx->fencePool, cmdState->complete);
        ReleaseResource(cmdState->cmdSource, cmdState->cmd);
        cmdState->onRetire(ctx, cmdState->resources);
    }

    ctx->vkCtx.vkScratch.top = save;
    return status == VK_SUCCESS;
}
void RetireCmdState(RenderContext* ctx, CmdState* cmdState) {

    VK_CALL(ctx->vkCtx.vkScratch, vkWaitForFences, ctx->vkCtx.logicalDevice, 1, &cmdState->complete, true, ~u64(0));
    ReleaseResource(&ctx->fencePool, cmdState->complete);
    ReleaseResource(cmdState->cmdSource, cmdState->cmd);
    cmdState->onRetire(ctx, cmdState->resources);
}


u32 RetireInFlightCmd(RenderContext* ctx, u32 count, CmdState* cmds) {

    for(u32 i = 0; i < count; i++) {
        bool retired = TryRetireCmdState(ctx, cmds + i);
        count -= retired;
        auto cpyIndex = count * retired + i * !retired;
        cmds[i] = cmds[cpyIndex];
        i -= retired;
    }
    if(ctx->descriptorUpdates.count) {
        VK_CALL(ctx->vkCtx.vkScratch, vkUpdateDescriptorSets, ctx->vkCtx.logicalDevice, ctx->descriptorUpdates.count, ctx->descriptorUpdates.write, 0, nullptr);
        ctx->descriptorUpdates.count = 0;
    }
    return count;
}
CmdState AcquireTransferResources(RenderContext* ctx) {

    CmdState ret{};
    ret.complete = AcquireResource(&ctx->fencePool);
    ret.cmd = AcquireResource(&ctx->transferCmdPool);
    ret.cmdSource = &ctx->transferCmdPool;

    return ret;
}
CmdState AcquireGraphicsResources(RenderContext* ctx) {

    CmdState ret{};
    ret.complete = AcquireResource(&ctx->fencePool);
    ret.cmd = AcquireResource(&ctx->graphicsCmdPool);
    ret.cmdSource = &ctx->graphicsCmdPool;

    return ret;
}
void IssueCmdState(RenderContext* ctx, CmdState* cmd, u32 waitCount, VkSemaphore* wait, VkPipelineStageFlags* stages, u32 signalCount, VkSemaphore* signal) {

    VkSubmitInfo sumbitInfo{};
    sumbitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sumbitInfo.commandBufferCount = 1;
    sumbitInfo.pCommandBuffers = &cmd->cmd;
    sumbitInfo.signalSemaphoreCount = signalCount;
    sumbitInfo.waitSemaphoreCount = waitCount;
    sumbitInfo.pWaitSemaphores = wait;
    sumbitInfo.pSignalSemaphores = signal;
    sumbitInfo.pWaitDstStageMask = stages;

    VK_CALL(ctx->vkCtx.vkScratch, vkResetFences, ctx->vkCtx.logicalDevice, 1, &cmd->complete);
    VK_CALL(ctx->vkCtx.vkScratch, vkQueueSubmit, ctx->graphicsQueue, 1, &sumbitInfo, cmd->complete);
}
void IssuePresentImg(RenderContext* ctx, u32 imgIndex, VkSemaphore wait) {
    
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &wait;
    presentInfo.swapchainCount = 1;
    presentInfo.pImageIndices = &imgIndex;
    presentInfo.pSwapchains = &ctx->swapChain;

    vkQueuePresentKHR(ctx->presentQueue, &presentInfo);
}


bool AreTransferResourcesReady(RenderContext* ctx) {

    auto available = IsResourceAvailable(&ctx->fencePool);
    available &= IsResourceAvailable(&ctx->transferCmdPool);
    available &= IsResourceAvailable(&ctx->semaphorePool);
    return available;
}
bool AreRenderResourcesReady(RenderContext* ctx) {

    auto available = IsResourceAvailable(&ctx->descriptorSetPool);
    available &= IsResourceAvailable(&ctx->fencePool);
    available &= IsResourceAvailable(&ctx->graphicsCmdPool);
    available &= IsResourceAvailable(&ctx->semaphorePool);
    return available;
}

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

void OnDrawFlatRetire(RenderContext* ctx, void* resources) {

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
void OnDrawFlatRetireCapture(RenderContext* ctx, void* resources) {

    char fileName[256];
    auto resourceCount = Mem<u32>(resources);
    byte* it = (byte*)(resources + 4);
    for(u32 i = 0; i < resourceCount; i++) {
        auto res = (DrawflatCmdResources*)it;
        ReleaseResource(&ctx->semaphorePool, res->imgAcquired);
        ReleaseResource(&ctx->semaphorePool, res->renderCompleted);
        ReleaseResource(&ctx->descriptorSetPool, res->descriptor);
        
        auto size = res->allocations[2].size;
        auto texels = (Pixel*)res->allocations[2].ptr;

        u32 imgSize = ctx->width * ctx->height * 4;
        Pixel* dst = nullptr;
        while(!dst) {
            dst = (Pixel*)circular_get_write_ptr(&res->engine->threadComm->images, imgSize);
        }

        for(u32 i = 0; i < size / sizeof(Pixel); i++) {
            dst[i].r = texels[i].b;
            dst[i].g = texels[i].g;
            dst[i].b = texels[i].r;
            dst[i].a = texels[i].a;
        }
        circular_advance_write(&res->engine->threadComm->images, imgSize);

        for(u32 k = 0; k < res->allocationCount; k++) {
            linear_free(&ctx->uploadMemory, res->allocations[k].ptr, res->allocations[k].size);
        }

        it = it + sizeof(DrawflatCmdResources) + res->allocationCount * sizeof(UploadAllocation);
    }
}

void RecordDrawFlatCmd(RenderContext* ctx, CmdState* cmdState, VkDescriptorSet descriptors, u32 imgIndex, u32 drawCount, DrawInfo* draws, u64 instanceOffset) {

    u64 off = instanceOffset;
    for(u32 i = 0; i < drawCount; i++) {
        off += draws[i].instanceCount * sizeof(InstanceInfo);
    }

    VkClearValue clearColor[2]{};
    clearColor[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearColor[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = ctx->renderPass;
    renderPassInfo.framebuffer = ctx->swapChainFrames[imgIndex].frameBuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {ctx->width, ctx->height};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearColor;

    VK_CALL(ctx->vkCtx.vkScratch, vkCmdBeginRenderPass,     cmdState->cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdBindPipeline,        cmdState->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->graphicsPipeline);
    VkViewport viewport;
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (f32)ctx->width;
    viewport.height = (f32)ctx->height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdSetViewport, cmdState->cmd, 0, 1, &viewport);
    VkRect2D scissor;
    scissor.offset = {0,0};
    scissor.extent = {ctx->width, ctx->height};
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdSetScissor, cmdState->cmd, 0, 1, &scissor);

    VkDescriptorSet boundSets[2] = {descriptors, ctx->textureDescriptors};
    VK_CALL(ctx->vkCtx.vkScratch, vkCmdBindDescriptorSets,  cmdState->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipelineLayout, 0, 2, boundSets, 0, nullptr);

    VkBuffer buffers[2] = {ctx->deviceBuffer, ctx->deviceBuffer};
    u64 offsets[2] = {0, instanceOffset};

    for(u32 i = 0; i < drawCount; i++) {
        offsets[0] = draws[i].model.vertexOffset;

        VK_CALL(ctx->vkCtx.vkScratch, vkCmdBindVertexBuffers,   cmdState->cmd, 0, 2, buffers, offsets);
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdBindIndexBuffer,     cmdState->cmd, ctx->deviceBuffer, draws[i].model.indexOffset, VK_INDEX_TYPE_UINT32);
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdDrawIndexed,         cmdState->cmd, draws[i].model.indexCount, draws[i].instanceCount, 0, 0, 0);

        offsets[1] += draws[i].instanceCount * sizeof(InstanceInfo);
    }

    VK_CALL(ctx->vkCtx.vkScratch, vkCmdEndRenderPass, cmdState->cmd);
}

u32 IssueSwapChainAcquire(EngineState* state, RenderContext* ctx, VkSemaphore signalSemaphore, VkFence signalFence) {
    
    u32 imgIndex;
    auto s = ctx->vkCtx.vkScratch;

    img_acquire:
    VkResult imgAcquireResult = vkAcquireNextImageKHR(ctx->vkCtx.logicalDevice, ctx->swapChain, ~u64(0), signalSemaphore, signalFence, &imgIndex);
    if(imgAcquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        
        i32 w,h;
        GLFW_CALL(glfwGetFramebufferSize(ctx->glfwHandle, &w, &h));
        ctx->width = w;
        ctx->height = h;
        f32 ratio = (f32)w / (f32)h;
        state->projection = ComputePerspectiveMat4(ToRadian(90.0f), ratio, 0.1f, 100.0f);
        RecreateSwapChain(ctx, w, h);
        goto img_acquire;
    }
    ctx->vkCtx.vkScratch = s;

    return imgIndex;
}


void ScreenCapture(EngineState* state, RenderContext* ctx, u32* screenShot, u32* counter) {

    GLFW_CALL(auto mKey = glfwGetKey(ctx->glfwHandle, GLFW_KEY_M));
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
}
void DestroyTexture(VkContext* ctx, VkTextureInfo texture) {
    VK_CALL(ctx->vkScratch, vkDestroyImageView, ctx->logicalDevice, texture.view, &ctx->vkAllocator);
    VK_CALL(ctx->vkScratch, vkDestroyImage, ctx->logicalDevice, texture.img, &ctx->vkAllocator);
}

void* ThreadLoop(void* mem) {

    auto block = (ThreadCommBlock*)mem;
    u32 counter = 0;
    char name[256];

    while(block->run) {

        std::this_thread::sleep_for(milli_second_t(5));

        u32 imgSize = block->x * block->y * sizeof(Pixel);
        auto size = circular_read_size(&block->images, imgSize);
        while(size >= imgSize) {

            auto img = (Pixel*)circular_get_read_ptr(&block->images, imgSize);
            local_print((byte*)name, 256, "sus", "../bin/screen_shoot", counter++, ".png");
            stbi_write_png(name, block->x, block->y, STBI_rgb_alpha, img, block->x * STBI_rgb_alpha);

            circular_advance_read(&block->images, imgSize);
            size = circular_read_size(&block->images, imgSize);
        }

    }
}

void global_io_flush_wrapper(void* user) {
    global_io_flush();
}
void global_print_wrapper(void* user, const char* format ...) {
    
    va_list args;
    va_start(args, format);
    auto end = print_fn_v(io.base+io.top, linear_allocator_free_size(&io), format, args);
    va_end(args);

    auto top = (byte*)linear_allocator_top(&io);
    if( (end - io.base) >= io.cap) {
        global_io_flush();
        top = (byte*)linear_allocator_top(&io);

        va_start(args, format);
        end = print_fn_v(top, linear_allocator_free_size(&io), format, args);
        va_end(args);

        ASSERT(end != top);
    }
    linear_allocate(&io, end-top);
}


i32 main(i32 argc, const char** argv) {

    auto mem = init_global_state(0, Megabyte(256), 512);

    auto size = ReadFile("../general.jpg", mem);
    auto alloc = make_linear_allocator(((byte*)mem + size), Megabyte(256) - size);
    ParseJFIFMemory(mem, size, &alloc);

    exit(0);
    glfwInit();

    ThreadCommBlock comms;
    comms.run = true;
    comms.images = make_ring_buffer(mem, Megabyte(32));
    comms.x = 300;
    comms.y = 300;
    mem += Megabyte(32);

    EngineState state;
    RenderContext ctx;
    MakeRenderContext(&ctx, mem , Megabyte(64));

    i32 monitorCount = 0;
    GLFW_CALL(state.primary = glfwGetMonitors(&monitorCount)[0]); 
    GLFW_CALL(glfwSetCursorPos(ctx.glfwHandle, (f64)ctx.width*0.5f, (f64)ctx.height * 0.5));
    state.camera.position = {2,2,2};
    state.camera.direction = normalize(vec<f32,3>{0,0,0} - state.camera.position);
    state.camera.vel = {0,0,0};

    state.threadComm = &comms;
    state.ctx = &ctx;
    state.time = 0;
    state.delta = 0;
    state.fullscreen = false;
    state.projection = ComputePerspectiveMat4(ToRadian(90.0f), ctx.width / (f32)ctx.height, 0.1f, 100.0f);

    u32 screenShot = 0;

    auto texture = UploadVkTexture(&ctx, {});
    auto texID = RegisterTexture(&ctx, texture, ctx.textureSampler, ctx.textureDescriptors);

    auto info = LoadOBJ(ctx.uploadMemory.base, (byte*)linear_top(&ctx.uploadMemory), "../res/rooom.obj");
    auto model = UploadModel(&ctx, info);

    auto block = allocate_gpu_block(&ctx.gpuAllocator, sizeof(InstanceInfo) * 512, sizeof(InstanceInfo));

    DrawInfo draw;
    draw.instanceCount = 1;
    draw.model = model;

    CmdState cmds[2];
    u32 inFlightCmds = 0;

    byte* mem_[256];
    auto ringAllocaotor = make_circular_allocator(mem_, 256);

    pthread_t thread0;
    pthread_create(&thread0, nullptr, ThreadLoop, &comms);

    while(!glfwWindowShouldClose(ctx.glfwHandle)) {
        ScopedAllocator save(&ctx.vkCtx.vkScratch);

        screenShot++;
        std::this_thread::sleep_for(milli_second_t(1));
        Update(&state);

        GLFW_CALL(auto mKey = glfwGetKey(ctx.glfwHandle, GLFW_KEY_M) && screenShot > 10);

        inFlightCmds = RetireInFlightCmd(&ctx, inFlightCmds, cmds);
        if(AreRenderResourcesReady(&ctx) && inFlightCmds == 0) {

            auto resourcesMem = circular_allocate(&ringAllocaotor, sizeof(u32) + sizeof(DrawflatCmdResources) + sizeof(UploadAllocation) * (mKey + 2));
            Mem<u32>(resourcesMem) = 1;
            auto resources = (DrawflatCmdResources*)(resourcesMem + 4);

            resources->imgAcquired = AcquireResource(&ctx.semaphorePool);
            auto img = IssueSwapChainAcquire(&state, &ctx, resources->imgAcquired, nullptr);
            resources->descriptor = AcquireResource(&ctx.descriptorSetPool);
            resources->renderCompleted = AcquireResource(&ctx.semaphorePool);
            resources->allocationCount = 2 + mKey;
            resources->allocations[0].ptr = linear_alloc(&ctx.uploadMemory, sizeof(CommonParams));
            resources->allocations[0].size = sizeof(CommonParams);
            resources->allocations[1].ptr = linear_alloc(&ctx.uploadMemory, sizeof(InstanceInfo) * 100);
            resources->allocations[1].size = sizeof(InstanceInfo) * 100;

            auto renderArgs = (CommonParams*)resources->allocations[0].ptr;
            renderArgs->projectionViewMatrix = state.projection * LookAt(state.camera.position, state.camera.position + state.camera.direction);

            auto instances = (InstanceInfo*)resources->allocations[1].ptr;
            for(u32 i = 0; i < 10; i++) {
                for(u32 k = 0; k < 10; k++) {

                    instances[i * 10 + k].textureIndex = texID;
                    instances[i * 10 + k].transform = ComputeRotarionXMat4(0);
                    instances[i * 10 + k].translation = {(f32)i * 2, 0, (f32)k * 2};
                }
            }

            auto cmd = AcquireGraphicsResources(&ctx);
            cmd.resources = resourcesMem;
            cmd.onRetire = OnDrawFlatRetire;
            VkBeginCmd(&ctx.vkCtx, cmd.cmd);

            VkBufferCopy copy{};
            copy.size = sizeof(CommonParams);
            copy.srcOffset = (byte*)resources->allocations[0].ptr - ctx.uploadMemory.base;
            copy.dstOffset = resources->descriptor.offset;
            VK_CALL(ctx.vkCtx.vkScratch, vkCmdCopyBuffer, cmd.cmd, ctx.hostBuffer, ctx.deviceBuffer, 1, &copy);

            VkBufferMemoryBarrier descriptorBarrier{};
            descriptorBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            descriptorBarrier.buffer = ctx.deviceBuffer;
            descriptorBarrier.offset = copy.dstOffset;
            descriptorBarrier.size = copy.size;
            descriptorBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            descriptorBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            descriptorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            descriptorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            VK_CALL(ctx.vkCtx.vkScratch, vkCmdPipelineBarrier, cmd.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &descriptorBarrier, 0, nullptr);

            VkBufferCopy instanceCpy{};
            instanceCpy.size = resources->allocations[1].size;
            instanceCpy.srcOffset = (byte*)resources->allocations[1].ptr - ctx.uploadMemory.base;
            instanceCpy.dstOffset = block.offset;
            VK_CALL(ctx.vkCtx.vkScratch, vkCmdCopyBuffer, cmd.cmd, ctx.hostBuffer, ctx.deviceBuffer, 1, &instanceCpy);

            VkBufferMemoryBarrier instanceBarrier{};
            instanceBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            instanceBarrier.buffer = ctx.deviceBuffer;
            instanceBarrier.offset = instanceCpy.dstOffset;
            instanceBarrier.size = instanceCpy.size;
            instanceBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            instanceBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            instanceBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            instanceBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            VK_CALL(ctx.vkCtx.vkScratch, vkCmdPipelineBarrier, cmd.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &instanceBarrier, 0, nullptr);

            RecordDrawFlatCmd(&ctx, &cmd, resources->descriptor.set, img, 1, &draw, block.offset);

            if(mKey && screenShot > 10) {
                screenShot = 0;
                cmd.onRetire = OnDrawFlatRetireCapture;
                resources->engine = &state;
                VkImageMemoryBarrier imgBarrier{};
                imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imgBarrier.image = ctx.swapChainFrames[img].colorImg;
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
                VK_CALL(ctx.vkCtx.vkScratch, vkCmdPipelineBarrier, cmd.cmd, srcStage, dstStage, 0,  0, nullptr, 0, nullptr, 1, &imgBarrier);

                resources->allocations[2].size = ctx.width * ctx.height * 4;
                resources->allocations[2].ptr = linear_alloc(&ctx.uploadMemory, ctx.width * ctx.height * 4);

                VkBufferImageCopy region{};
                region.imageExtent = {ctx.width, ctx.height, 1};
                region.imageOffset = {0,0,0};
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.layerCount = 1;
                region.bufferImageHeight = ctx.height;
                region.bufferOffset = (byte*)resources->allocations[2].ptr - ctx.uploadMemory.base;
                region.bufferRowLength = ctx.width;
                VK_CALL(ctx.vkCtx.vkScratch, vkCmdCopyImageToBuffer, cmd.cmd, ctx.swapChainFrames[img].colorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ctx.hostBuffer, 1, &region);

                imgBarrier = {};
                imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imgBarrier.image = ctx.swapChainFrames[img].colorImg;
                imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                imgBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                imgBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imgBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                imgBarrier.subresourceRange.layerCount = 1;
                imgBarrier.subresourceRange.levelCount = 1;
                imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                imgBarrier.dstAccessMask = 0;

                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                VK_CALL(ctx.vkCtx.vkScratch, vkCmdPipelineBarrier, cmd.cmd, srcStage, dstStage, 0,  0, nullptr, 0, nullptr, 1, &imgBarrier);
            }

            VK_CALL(ctx.vkCtx.vkScratch, vkEndCommandBuffer, cmd.cmd);

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            IssueCmdState(&ctx, &cmd, 1, &resources->imgAcquired, &waitStage, 1, &resources->renderCompleted);
            IssuePresentImg(&ctx, img, resources->renderCompleted);
            cmds[inFlightCmds++] = cmd;
        }
        
        global_print("fcuc", (f64)ctx.uploadMemory.top / (f64)ctx.uploadMemory.cap, ' ', ringAllocaotor.head, '\n');
        VkFlushLog(&ctx.ioLogBuffer);
        global_io_flush();
    }

    comms.run = false;

    VK_CALL(ctx.vkCtx.vkScratch, vkQueueWaitIdle, ctx.graphicsQueue);
    DestroyTexture(&ctx.vkCtx, texture);
    free_gpu_block(&ctx.gpuAllocator, texture.memory);

    for(u32 i = 0; i < inFlightCmds; i++) {
        RetireCmdState(&ctx, cmds + i);
    }
    DestroyRenderContext(&ctx);
    glfwTerminate();

    void* ret;
    pthread_join(thread0, &ret);

    return 0;
}
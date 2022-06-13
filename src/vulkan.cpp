#define VULKAN_DEBUG 1
#include <vulkan.h>
#include <debug.h>
#include <common.h>

#include <malloc.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

const char* VALIDATION_LAYERS[] = {
    "VK_LAYER_KHRONOS_validation",
};
const char* DEVICE_EXTENSIONS[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_MAINTENANCE3_EXTENSION_NAME,
};
const char* INSTANCE_EXTENSIONS[] = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_XCB_SURFACE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
};


extern const byte flat_spv[760];
extern const byte vertex2d_spv[2308];
const VkViewport      global_debug_view_port      = {0,0, 640,480 ,0,0};
const VkRect2D        global_debug_scissor        = { {0,0}, {640,480} };
extern const DescriptorSetLayoutInfo global_render_info_descriptor_layout_info = {

    1,
    (u32[1]) {
        sizeof(GlobalRenderParams)
    },
    (VkDescriptorBindingFlags[1]) {
        0
    },
    (VkDescriptorSetLayoutBinding[1]) {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, 0},
    },
};
extern const DescriptorSetLayoutInfo global_textures_descriptor_layout_info = {

    1,
    (u32[1]) {
        0
    },
    (VkDescriptorBindingFlags[1]) {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT,
    },
    (VkDescriptorSetLayoutBinding[1]) {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512, VK_SHADER_STAGE_FRAGMENT_BIT, 0}
    },
};
extern const PipelineDescriptor global_debug_flat2d_pipeline = {

    VK_PIPELINE_BIND_POINT_GRAPHICS,
    {flat_spv, sizeof(flat_spv)}, {vertex2d_spv, sizeof(vertex2d_spv)},
    {
        2,
        1,
         (VkVertexInputAttributeDescription[2]) {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex2, pos)},
            {1, 0, VK_FORMAT_R32_UINT,      offsetof(Vertex2, col)}
        },
         (VkVertexInputBindingDescription[1]) {
            {0, sizeof(Vertex2), VK_VERTEX_INPUT_RATE_VERTEX}
        }
    },

    {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, 0,0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false},
    {},
    {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,      0,0, 1, &global_debug_view_port, 1, &global_debug_scissor},
    {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, 0,0, false, false, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, false, 0,0,0,1.0},
    {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,   0,0, VK_SAMPLE_COUNT_1_BIT, false, 0,0,0,0},
    {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, 0,0, 0,0,VK_COMPARE_OP_NEVER,0,0, {}, {}, 0,1.0},
    {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,   0,0, 0,VK_LOGIC_OP_CLEAR, 1, (VkPipelineColorBlendAttachmentState[1]){
        0,VK_BLEND_FACTOR_ZERO,VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ZERO,VK_BLEND_FACTOR_ZERO,VK_BLEND_OP_ADD,
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    }, {0.0, 0.0, 0.0, 0.0}},
    {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,       0,0, 2, (VkDynamicState[2]){VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR} },
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


    ASSERT(user);
    auto ctx = (VkCoreContext*)user;
    if(scope == VK_SYSTEM_ALLOCATION_SCOPE_COMMAND) {
        auto mem = linear_aligned_allocate(&ctx->vkScratch, size, align);
        ASSERT(mem);
        return mem;
    }

    auto mem = aligned_alloc(align, size);
    ASSERT(mem);
    return mem;
}
void vkLocalFree(void* user, void* mem) {

    ASSERT(user);
    auto ctx = (VkCoreContext*)user;
    if(!mem || (byte*)mem <= ctx->vkScratch.base + ctx->vkScratch.cap) {
        return;
    }
    free(mem);
}
void* vkLocalRealloc(void* user, void* og, size_t size, size_t alignment, VkSystemAllocationScope scope) {

    if(scope == VK_SYSTEM_ALLOCATION_SCOPE_COMMAND) {
        ASSERT(false);
    }

    auto fresh = (byte*)aligned_alloc(alignment, size);
    if(og) {

        auto prevSize = malloc_usable_size(og);
        memcpy(fresh, og, prevSize);
        free(og);
    }
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


VkBool32 VKAPI_PTR DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {

    constexpr const char* SEVERITY_STR[] = {
        "VERBOSE",
        "INFO",
        "WARNING",
        "ERROR"
    };
    auto ctx = (VkCoreContext*)pUserData;
    if(ctx->logSeverity & messageSeverity) {
    }
    VkLog(&ctx->ioLog, "sscsc", "[vulkan validation layer info] severity ", SEVERITY_STR[(i32)f32_log(messageSeverity, 16)], ' ', pCallbackData->pMessage, '\n');
    VkFlushLog(&ctx->ioLog);
    return VK_FALSE;
}

VkDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT(VkCoreContext* ctx, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo) {

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(ctx->vkInstance, "vkCreateDebugUtilsMessengerEXT");
    ASSERT(func);

    VkDebugUtilsMessengerEXT pDebugMessenger;
    VK_CALL(ctx->vkScratch, func, ctx->vkInstance, pCreateInfo, &ctx->vkAllocator, &pDebugMessenger);

    return pDebugMessenger;
}
void DestroyDebugUtilsMessengerEXT(VkCoreContext* ctx, VkDebugUtilsMessengerEXT debugMessenger) {
    
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(ctx->vkInstance, "vkDestroyDebugUtilsMessengerEXT");
    ASSERT(func);
    VK_CALL(ctx->vkScratch, func, ctx->vkInstance, debugMessenger, &ctx->vkAllocator);
}

f32 DeviceScore(VkCoreContext* ctx, VkPhysicalDevice device) {

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

const char* GetVkResultEnumStr(VkResult result) {

    switch(result) {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_NOT_READY:
        return "VK_NOT_READY";
    case VK_TIMEOUT:
        return "VK_TIMEOUT";
    case VK_EVENT_SET:
        return "VK_EVENT_SET";
    case VK_EVENT_RESET:
        return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
        return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
        return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
        return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
        return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
        return "VK_ERROR_UNKNOWN";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
        return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
        return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION:
        return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
        return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case VK_ERROR_SURFACE_LOST_KHR:
        return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
        return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR:
        return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
        return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
        return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT:
        return "VK_ERROR_VALIDATION_FAILED_EXT";
    case VK_ERROR_INVALID_SHADER_NV:
        return "VK_ERROR_INVALID_SHADER_NV";
    case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
        return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
    case VK_ERROR_NOT_PERMITTED_EXT:
        return "VK_ERROR_NOT_PERMITTED_EXT";
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
        return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    }

    return nullptr;
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
void PrintQueueFamilies(VkPhysicalDevice device) {

    u32 queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    VkQueueFamilyProperties queueFamilies[queueFamilyCount];
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);

    for(u32 i = 0; i < queueFamilyCount; i++) {

        global_print("susus", "queue family [", i, "] queue count [", queueFamilies[i].queueCount, "] ");
        PrintQueueFamilyFlags(queueFamilies[i].queueFlags);
        global_print("c", '\n');
    }
}

QueueFamilies GetQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {

    u32 queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    VkQueueFamilyProperties queueFamilies[queueFamilyCount];
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);

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
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if(presentSupport) {
            families.presentFamily = i;
            break;
        }
    }

    return families;
}

void PrintAvailableDeviceExt(VkPhysicalDevice device) {

    u32 extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    VkExtensionProperties availableExtensions[extensionCount];
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions);

    for(u32 i = 0; i < extensionCount; i++) {
        global_print("ucsc", i, ' ', availableExtensions[i].extensionName, '\n');
    }
    global_io_flush();
}

bool CheckDeviceExtensionSupport(VkCoreContext* ctx, VkPhysicalDevice device, const char** deviceExtensions, u32 deviceExtensionsCount) {
 
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
        if(!found) {
            return false;
        }
    }

    return true;
}

void PrintDeviceProperties(VkPhysicalDevice device) {

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(device , &physicalDeviceProperties);

    const char* DEVICE_TYPE_STR[] = {
        "VK_PHYSICAL_DEVICE_TYPE_OTHER",
        "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU",
        "VK_PHYSICAL_DEVICE_TYPE_CPU",
    };
    global_print("suc", "apiVersion ",           physicalDeviceProperties.apiVersion,                   '\n');
    global_print("suc", "deviceID ",             physicalDeviceProperties.deviceID,                     '\n');
    global_print("ssc", "deviceName ",           physicalDeviceProperties.deviceName,                   '\n');
    if(physicalDeviceProperties.deviceType > -1 && physicalDeviceProperties.deviceType < SIZE_OF_ARRAY(DEVICE_TYPE_STR)) {
        global_print("ssc", "deviceType ",           DEVICE_TYPE_STR[physicalDeviceProperties.deviceType],  '\n');
    }
    else {
        global_print("ssc", "deviceType ",           "uknown device type",  '\n');
    }
    global_print("suc", "driverVersion ",        physicalDeviceProperties.driverVersion,                '\n');
    global_print("suc", "pipelineCacheUUID ",    physicalDeviceProperties.pipelineCacheUUID,            '\n');
    global_print("suc", "vendorID ",             physicalDeviceProperties.vendorID,                     '\n');
    global_io_flush();
}
bool IsDeviceSuitable(VkCoreContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface, const char** requiredDeviceExt, u32 count) {

    auto family = GetQueueFamilies(device, surface);
    auto extSupport = CheckDeviceExtensionSupport(ctx, device, requiredDeviceExt, count);

    if(!family.graphicsFamily == ~u32(0) || family.presentFamily == ~u32(0) || !extSupport) return false;

    u32 formatCount;
    u32 presentModeCount;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, nullptr);
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfacePresentModesKHR, device, surface, &presentModeCount, nullptr);

    return formatCount && presentModeCount;
}

u32 GetGPUs(VkCoreContext* core, VkPhysicalDevice* dst) {

    u32 deviceCount = 0;
    VK_CALL(core->vkScratch, vkEnumeratePhysicalDevices, core->vkInstance, &deviceCount, nullptr);
    if(deviceCount == 0) {
        return 0;
    }

    VK_CALL(core->vkScratch, vkEnumeratePhysicalDevices, core->vkInstance, &deviceCount, dst);
    return deviceCount;
}
VkPhysicalDevice PickPhysicalDevice(VkCoreContext* ctx, VkSurfaceKHR surface, u32 gpuCount, VkPhysicalDevice* gpus, const char** requiredDeviceExt, u32 count) {

    f32 max = 0;
    u32 device = ~u32(0);
    for(u32 i = 0; i < gpuCount; i++) {

        if(!IsDeviceSuitable(ctx, gpus[i], surface, requiredDeviceExt, count)) continue;
        f32 score = DeviceScore(ctx, gpus[i]);
        device = score > max ? i : device;
        max = score > max ? score : max;
    }
    ASSERT(device != ~u32(0));

    return gpus[device];
}



VkDevice CreateLogicalDevice(VkCoreContext* ctx, QueueFamilies families, VkPhysicalDevice physicalDevice) {
   
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

    VkDeviceCreateInfo logicalDeviceCreateInfo{};
    logicalDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    logicalDeviceCreateInfo.pQueueCreateInfos = queueCreateInfos;
    logicalDeviceCreateInfo.queueCreateInfoCount = uniqueFamilyCount;
    logicalDeviceCreateInfo.pNext = &deviceFeatures2;
    
    logicalDeviceCreateInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS;
    logicalDeviceCreateInfo.enabledExtensionCount = SIZE_OF_ARRAY(DEVICE_EXTENSIONS);

    if constexpr (ENABLE_VALIDATION_LAYER) {
        logicalDeviceCreateInfo.enabledLayerCount = SIZE_OF_ARRAY(VALIDATION_LAYERS);
        logicalDeviceCreateInfo.ppEnabledLayerNames = VALIDATION_LAYERS;
    }
    
    VkDevice device;
    VK_CALL(ctx->vkScratch, vkCreateDevice, physicalDevice, &logicalDeviceCreateInfo, &ctx->vkAllocator, &device);

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
        actualExtent.width = Clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = Clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }
}
VkSurfaceFormatKHR GetSurfaceFormat(VkCoreContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface) {

    u32 formatCount;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, nullptr);

    VkSurfaceFormatKHR formats[formatCount];
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, formats);

    return ChooseSwapSurfaceFormat(formats, formatCount);
}
SwapChainResult CreateSwapChain(VkCoreContext* ctx, u32 frameCount, VkPhysicalDevice device, VkDevice logicalDevice, VkSurfaceKHR surface, vec<u32,2> dims, QueueFamilies families, VkSwapchainKHR oldChain) {

    u32 formatCount;
    u32 presentModeCount;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, nullptr);
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfacePresentModesKHR, device, surface, &presentModeCount, nullptr);

    VkSurfaceFormatKHR formats[formatCount];
    VkPresentModeKHR presentModes[presentModeCount];
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceFormatsKHR, device, surface, &formatCount, formats);
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfacePresentModesKHR, device, surface, &presentModeCount, presentModes);

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


    SwapChainResult ret;
    ret.format = surfaceFormat.format;

    VkSwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainCreateInfo.surface = surface;
    swapChainCreateInfo.imageFormat = surfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
    
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

    swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainCreateInfo.presentMode = presentMode;
    swapChainCreateInfo.clipped = VK_TRUE;
    swapChainCreateInfo.oldSwapchain = oldChain;

    VkSurfaceCapabilitiesKHR capabilities;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceSurfaceCapabilitiesKHR, device, surface, &capabilities);
    swapChainCreateInfo.imageExtent = ChooseSwapExtent(capabilities, dims.x, dims.y);


    frameCount = capabilities.minImageCount + 1;
    if(capabilities.maxImageCount > 0 && frameCount > capabilities.maxImageCount) {
        frameCount = capabilities.maxImageCount;
    }

    swapChainCreateInfo.minImageCount = frameCount;
    swapChainCreateInfo.preTransform = capabilities.currentTransform;
    VK_CALL(ctx->vkScratch, vkCreateSwapchainKHR, logicalDevice, &swapChainCreateInfo, &ctx->vkAllocator, &ret.swapChain);

    ret.dims = {swapChainCreateInfo.imageExtent.width, swapChainCreateInfo.imageExtent.height};
    return ret;
}
VkFormat FindSupportedFormat(VkCoreContext* ctx, VkPhysicalDevice device, u32 count, VkFormat* candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

    for (u32 i = 0; i < count; i++) {
        VkFormatProperties props;
        VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceFormatProperties, device, candidates[i], &props);

        bool cond = (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) ||
                    (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features);
        if(cond) {
            return candidates[i];
        }
    }

    return VkFormat{};
}
VkFormat FindDepthFormat(VkCoreContext* ctx, VkPhysicalDevice device) {
    VkFormat formats[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    return FindSupportedFormat(ctx, device, SIZE_OF_ARRAY(formats), formats, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}


VkImage CreateImg2D(VkCoreContext* ctx, VkDevice logicalDevice, vec<u32,2> dims, u32* families, u32 familyCount, VkFormat format, VkImageUsageFlags usage) {

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
    imgInfo.sharingMode = (familyCount > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE);

    imgInfo.pQueueFamilyIndices = families;
    imgInfo.queueFamilyIndexCount = familyCount;

    VkImage img;
    VK_CALL(ctx->vkScratch, vkCreateImage, logicalDevice, &imgInfo, &ctx->vkAllocator, &img);

    return img;
}

MemBlock BackImgMemory(VkCoreContext* ctx, VkDevice logicalDevice, VkDeviceMemory memory, GpuHeap* gpuAllocator, VkImage img) {

    VkMemoryRequirements memRequirements;
    VK_CALL(ctx->vkScratch, vkGetImageMemoryRequirements, logicalDevice, img, &memRequirements);

    auto block = allocate_gpu_block(gpuAllocator, memRequirements.size, memRequirements.alignment);
    VK_CALL(ctx->vkScratch, vkBindImageMemory, logicalDevice, img, memory, block.offset);

    return block;
}

void IssueGPUCopytoImage(VkCoreContext* core, VkCommandBuffer commandBuffer, VkTextureInfo dst, VkBuffer srcBuffer, u64 srcOffset) {

    VkBufferImageCopy region{};
    region.bufferOffset = srcOffset;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {dst.width, dst.height, 1};

    VK_CALL(core->vkScratch, vkCmdCopyBufferToImage, commandBuffer, srcBuffer, dst.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}
void RecordGPUTextureUpload(VkCoreContext* core, VkGPUContext* gpu, CmdState* transfer, CmdState* graphics, VkTextureInfo vkTex, ImageDescriptor img) {

    u32 imgSize = img.width * img.height * 4;
    auto texels = img.img;

    auto dst = (Pixel*)linear_alloc(&gpu->uploadMemory, imgSize);
    ASSERT(dst);
    memcpy(dst, texels, imgSize);

    CommandFreeHostAlloc* cmdFree;
    if(*((CpuCMDOp*)transfer->currentCmd) == CMD_FREE_HOST_ALLOC) {
        cmdFree = (CommandFreeHostAlloc*)transfer->currentCmd;
    }
    else {
        auto cmd = (CpuCommand*)transfer->currentCmd;
        auto end = (byte*)(cmd + 1) + cmd->len;
        cmdFree = (CommandFreeHostAlloc*)end;
        cmdFree->op = CMD_FREE_HOST_ALLOC;
        cmdFree->len = 0;
        transfer->currentCmd = cmdFree;
    }
    auto alloc = (Allocation*)((byte*)cmdFree->allocs + cmdFree->len);
    alloc->ptr = dst;
    alloc->size = imgSize;
    cmdFree->len += sizeof(Allocation);

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkImageMemoryBarrier toTransferDstOptimal{};
    toTransferDstOptimal.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDstOptimal.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDstOptimal.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDstOptimal.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstOptimal.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstOptimal.image = vkTex.img;
    toTransferDstOptimal.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferDstOptimal.subresourceRange.levelCount = 1;
    toTransferDstOptimal.subresourceRange.layerCount = 1;
    toTransferDstOptimal.srcAccessMask = 0;
    toTransferDstOptimal.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, transfer->cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &toTransferDstOptimal);

    VkBufferImageCopy region{};
    region.bufferOffset = (byte*)dst - gpu->uploadMemory.base;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {img.width,img.height, 1};
    VK_CALL(core->vkScratch, vkCmdCopyBufferToImage, transfer->cmd, gpu->hostBuffer, vkTex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier toShaderReadOPt{};
    toShaderReadOPt.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toShaderReadOPt.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShaderReadOPt.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShaderReadOPt.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShaderReadOPt.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShaderReadOPt.image = vkTex.img;
    toShaderReadOPt.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toShaderReadOPt.subresourceRange.baseMipLevel = 0;
    toShaderReadOPt.subresourceRange.levelCount = 1;
    toShaderReadOPt.subresourceRange.baseArrayLayer = 0;
    toShaderReadOPt.subresourceRange.layerCount = 1;
    toShaderReadOPt.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShaderReadOPt.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, graphics->cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &toShaderReadOPt);
}

void IssueGPUImageBarrier(VkCoreContext* core, VkCommandBuffer commandBuffer, VkTextureInfo img, ImageBarrierArgs args) {

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = args.oldLayout;
    barrier.newLayout = args.newLayout;
    barrier.srcQueueFamilyIndex = args.srcQueue;
    barrier.dstQueueFamilyIndex = args.dstQueue;
    barrier.image = img.img;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = args.flushCache;
    barrier.dstAccessMask = args.invalidateCache;

    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, commandBuffer, args.srcStage, args.dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

VkTextureInfo CreateVkTexture(VkCoreContext* core, VkGPUContext* gpu, ImageDescriptor img) {

    VkTextureInfo ret;
    ret.img = CreateImg2D(core, gpu->logicalDevice, {img.width,img.height}, 0, 0, (VkFormat)img.format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ret.memory = BackImgMemory(core, gpu->logicalDevice, gpu->deviceMemory, &gpu->gpuAllocator, ret.img);
    ret.width = img.width;
    ret.height = img.height;

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = ret.img;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = (VkFormat)img.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &viewInfo, &core->vkAllocator, &ret.view);

    return ret;
}

u32 CreateSwapChainFrames(VkCoreContext* core, VkGPUContext* gpu, SwapChainFrame* dst, VkRenderPass renderPass, VkSwapchainKHR swapChain, vec<u32, 2> dim, VkFormat depthFormat, VkFormat colorFormat) {

    u32 imageCount = 0;
    VK_CALL(core->vkScratch, vkGetSwapchainImagesKHR, gpu->logicalDevice, swapChain, &imageCount, nullptr);
    VkImage images[imageCount];
    VK_CALL(core->vkScratch, vkGetSwapchainImagesKHR, gpu->logicalDevice, swapChain, &imageCount, images);

    u32 families[2] = {gpu->families.graphicsFamily, gpu->families.presentFamily};
    auto count = get_unique(families, 2);
    for(u32 i = 0; i < imageCount; i++) {

        dst[i].colorImg = images[i];
        dst[i].depthImg.img = CreateImg2D(core, gpu->logicalDevice, dim, families, count, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        dst[i].depthImg.memory = BackImgMemory(core, gpu->logicalDevice, gpu->deviceMemory, &gpu->gpuAllocator, dst[i].depthImg.img);

        VkImageViewCreateInfo depthCreateInfo{};
        depthCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthCreateInfo.image    = dst[i].depthImg.img;
        depthCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthCreateInfo.format   = depthFormat;

        depthCreateInfo.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthCreateInfo.subresourceRange.baseMipLevel    = 0;
        depthCreateInfo.subresourceRange.levelCount      = 1;
        depthCreateInfo.subresourceRange.baseArrayLayer  = 0;
        depthCreateInfo.subresourceRange.layerCount      = 1;

        VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &depthCreateInfo, &core->vkAllocator, &dst[i].depthImgView);

        VkImageViewCreateInfo colorCreateInfo{};
        colorCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorCreateInfo.image    = dst[i].colorImg;
        colorCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorCreateInfo.format   = colorFormat;

        colorCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        colorCreateInfo.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
        colorCreateInfo.subresourceRange.baseMipLevel    = 0;
        colorCreateInfo.subresourceRange.levelCount      = 1;
        colorCreateInfo.subresourceRange.baseArrayLayer  = 0;
        colorCreateInfo.subresourceRange.layerCount      = 1;

        VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &colorCreateInfo, &core->vkAllocator, &dst[i].colorImgView);

        VkImageView attachments[] = {dst[i].colorImgView, dst[i].depthImgView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = SIZE_OF_ARRAY(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = dim.x;
        framebufferInfo.height = dim.y;
        framebufferInfo.layers = 1;

        VK_CALL(core->vkScratch , vkCreateFramebuffer, gpu->logicalDevice, &framebufferInfo, &core->vkAllocator, &dst[i].frameBuffer);
    }

    return imageCount;
}

VkShaderModule CreateShaderModule(VkCoreContext* ctx, VkDevice logicalDevice, const byte* source, u32 len) {

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = len;
    createInfo.pCode = (u32*)source;

    VkShaderModule shaderModule;
    VK_CALL(ctx->vkScratch, vkCreateShaderModule, logicalDevice, &createInfo, &ctx->vkAllocator, &shaderModule);

    return shaderModule;
}

VkDescriptorSetLayout CreateDescriptorSetLayouts(VkCoreContext* ctx, VkDevice logicalDevice, VkDescriptorSetLayoutBinding* bindings, VkDescriptorBindingFlags* flags, u32 count) {

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

VkRenderPass CreateRenderPass(VkCoreContext* ctx, VkDevice logicalDevice, VkFormat colorFormat, VkFormat depthFormat) {

    VkAttachmentDescription attachment[2]{};
    attachment[0].format = colorFormat;
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
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

    dependency.srcAccessMask = 0;
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
    VK_CALL(ctx->vkScratch, vkCreateRenderPass, logicalDevice, &renderPassInfo, &ctx->vkAllocator, &renderPass);

    return renderPass;
}

VkPipeline CreateGraphicsPipeline(VkCoreContext* ctx, VkDevice logicalDevice, VkPipelineLayout layout, VkRenderPass renderPass, VkPipelineVertexInputStateCreateInfo* input, const char* vertexPath, const char* fragPath, u32 width, u32 height) {

    auto allocSave = ctx->vkScratch;
    auto vertSource = (char*)linear_allocator_top(&ctx->vkScratch);
    u64 vertSourceSize = ReadFile(vertexPath, (byte*)vertSource, linear_allocator_free_size(&ctx->vkScratch));
    if(vertSourceSize == ~u64(0)) {
        ASSERT(false);
    }
    linear_allocate(&ctx->vkScratch, vertSourceSize);

    auto fragSource = (char*)linear_allocator_top(&ctx->vkScratch);
    u64 fragSourceSize = ReadFile(fragPath, (byte*)fragSource, linear_allocator_free_size(&ctx->vkScratch));
    if(fragSourceSize == ~u64(0)) {
        ASSERT(false);
    }
    linear_allocate(&ctx->vkScratch, fragSourceSize);

    auto vertexModule = CreateShaderModule(ctx, logicalDevice, (const byte*)vertSource, vertSourceSize);
    auto fragmentModule = CreateShaderModule(ctx, logicalDevice, (const byte*)fragSource, fragSourceSize);

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
    viewport.width = (f32)width;
    viewport.height = (f32)height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent.width = width;
    scissor.extent.height = height;

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
    pipelineInfo.pVertexInputState = input;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = layout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.pDepthStencilState = &depthStencilStateInfo;

    VkPipeline pipeline;
    VK_CALL(ctx->vkScratch, vkCreateGraphicsPipelines, logicalDevice, nullptr, 1, &pipelineInfo, &ctx->vkAllocator, &pipeline);

    VK_CALL(ctx->vkScratch, vkDestroyShaderModule, logicalDevice, vertexModule, &ctx->vkAllocator);
    VK_CALL(ctx->vkScratch, vkDestroyShaderModule, logicalDevice, fragmentModule, &ctx->vkAllocator);

    ctx->vkScratch = allocSave;

    return pipeline;
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
void PrintMemoryTypeBitsFlags(VkCoreContext* ctx, VkPhysicalDevice device, u32 typeFilter) {

    VkPhysicalDeviceMemoryProperties memProperties;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceMemoryProperties, device, &memProperties);

    for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {
        if( (typeFilter & (1 << i)) == (1 << i) ) {
            global_print("uc", i, '\n');
            PrintMemoryPropertyFlags(memProperties.memoryTypes[i].propertyFlags);
        }
    }
}
u32 MatchMemoryType(VkPhysicalDevice device, u32 typeFilter, VkMemoryPropertyFlags properties) {
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);

    for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {

        u32 bit = 1 << i;
        if( ((typeFilter & bit) == bit) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties) ) {
            return i;
        }
    }

    return ~u32(0);
}

u32 GetImportMemoryAlignment(VkCoreContext* ctx, VkPhysicalDevice device) {

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT al{};
    al.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 st{};
    st.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    st.pNext = &al;
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceProperties2, device, &st);
    return al.minImportedHostPointerAlignment;
}

VkDeviceMemory ImportMemory(VkCoreContext* ctx, VkDevice logicalDevice, void* mem, u32 size, u32 typeIndex) {

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
    VK_CALL(ctx->vkScratch, vkAllocateMemory, logicalDevice, &allocInfo, &ctx->vkAllocator, &memory);

    return memory;
}

VkDeviceMemory AllocateGPUMemory(VkCoreContext* ctx, VkDevice logicalDevice, u32 size, u32 typeIndex) {

    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = typeIndex;
    allocInfo.pNext = &flags;

    VkDeviceMemory memory = nullptr;
    VK_CALL(ctx->vkScratch, vkAllocateMemory, logicalDevice, &allocInfo, &ctx->vkAllocator, &memory);
    return memory;
}

struct VkBufferArgs {
    u32* queueFamilyIndicies;
    u32 queueFamilyCount;
    VkBufferUsageFlags usage;
    VkSharingMode sharing;
    bool externaMem;
};
VkBuffer MakeVkBuffer(VkCoreContext* ctx, VkDevice logicalDevice, u32 size, VkBufferArgs args) {

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
    VK_CALL(ctx->vkScratch, vkCreateBuffer, logicalDevice, &bufferInfo, &ctx->vkAllocator, &buffer);
    return buffer;
}

void IssueGPUMemCpy(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, LoadedInfo model) {

}

ModelDesciption UploadModel(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, LoadedInfo model) {

    ModelDesciption ret;
    ret.indexCount = model.indexSize / sizeof(u32);
    ret.vertexCount = model.vertexSize / sizeof(Vertex);

    VkCommandBuffer transferCmd;
    ASSERT(TryAcquireResource(&res->transferCmdPool, &transferCmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CALL(core->vkScratch, vkBeginCommandBuffer, transferCmd, &begin);

    auto block = allocate_gpu_block(&gpu->gpuAllocator, model.indexSize + model.vertexSize, 16);
    VkBufferCopy bufferInfo{};
    bufferInfo.size = block.size;
    bufferInfo.srcOffset = model.vertexOffset;
    bufferInfo.dstOffset = block.offset;

    ret.vertexOffset = block.offset;
    ret.indexOffset = block.offset + model.vertexSize;

    VK_CALL(core->vkScratch, vkCmdCopyBuffer, transferCmd, gpu->hostBuffer, gpu->deviceBuffer, 1, &bufferInfo);
    VK_CALL(core->vkScratch, vkEndCommandBuffer, transferCmd);

    VkFence finished;
    ASSERT(TryAcquireResource(&res->fencePool, &finished));
    VK_CALL(core->vkScratch, vkResetFences, gpu->logicalDevice, 1, &finished);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &transferCmd;
    
    VK_CALL(core->vkScratch, vkQueueSubmit, gpu->transferQueue, 1, &submit, finished);
    VK_CALL(core->vkScratch, vkWaitForFences, gpu->logicalDevice, 1, &finished, 1, ~u64(0));

    ReleaseResource(&res->fencePool, finished);
    ReleaseResource(&res->transferCmdPool, transferCmd);

    gpu->uploadMemory.top = model.vertexOffset;

    return ret;
}
void RecreateSwapChain(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkRenderPass renderPass, u32 width, u32 height) {

    VK_CALL(core->vkScratch, vkDeviceWaitIdle, gpu->logicalDevice);
    for(u32 i = 0; i < fbo->frameCount; i++) {
        VK_CALL(core->vkScratch, vkDestroyImageView, gpu->logicalDevice, fbo->frames[i].colorImgView, &core->vkAllocator);
        VK_CALL(core->vkScratch, vkDestroyImageView, gpu->logicalDevice, fbo->frames[i].depthImgView, &core->vkAllocator);
        VK_CALL(core->vkScratch, vkDestroyImage, gpu->logicalDevice, fbo->frames[i].depthImg.img, &core->vkAllocator);
        VK_CALL(core->vkScratch, vkDestroyFramebuffer, gpu->logicalDevice, fbo->frames[i].frameBuffer, &core->vkAllocator);
        free_gpu_block(&gpu->gpuAllocator, fbo->frames[i].depthImg.memory);
    }
    
    auto newChain = CreateSwapChain(core, fbo->frameCount, gpu->device, gpu->logicalDevice, fbo->surface, {width, height}, gpu->families, fbo->swapChain);
    fbo->swapChain = newChain.swapChain;
    fbo->fboColorFormat = newChain.format;
    fbo->width = newChain.dims.x;
    fbo->height = newChain.dims.y;
    fbo->frameCount = CreateSwapChainFrames(core, gpu, fbo->frames, renderPass, fbo->swapChain, {fbo->width, fbo->height}, fbo->depthFormat, fbo->fboColorFormat);

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


struct VkHeaps {
    CoalescingLinearAllocator uploadAllocator;

    VkDeviceMemory deviceMemory;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkBuffer hostBuffer;


    u64 deviceHeapGpuAddress;
    u64 hostHeapGpuAddress;
};

u64 GetGPUMemorySize(VkPhysicalDevice device) {

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);

    u64 ret = 0;
    for(u32 i = 0; i < memProperties.memoryHeapCount; i++) {
        if(memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT == VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            ret += memProperties.memoryHeaps[i].size;
        }
    }
    return ret;
}
VkHeaps CreateHeaps(VkCoreContext* ctx, VkPhysicalDevice device, VkDevice logicalDevice, QueueFamilies families, byte* localMemory, u32 localMemorySize, u32 gpuAllocatorSize) {

    VkHeaps heap{};

    auto aligment = GetImportMemoryAlignment(ctx, device);
    auto aligned = (byte*)align_pointer(localMemory, aligment);
    localMemorySize = (localMemorySize / aligment) * aligment;

    heap.uploadAllocator = make_coalescing_linear_allocator(aligned, localMemorySize);

    auto vkGetMemoryHostPointerPropertiesEXT_ = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(logicalDevice, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT prop{};
    prop.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    VK_CALL(ctx->vkScratch, vkGetMemoryHostPointerPropertiesEXT_, logicalDevice, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, localMemory, &prop);
    auto importMemoryType = MatchMemoryType(device, prop.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    heap.hostMemory = ImportMemory(ctx, logicalDevice, heap.uploadAllocator.base, heap.uploadAllocator.cap, importMemoryType);

    auto deviceMemoryType = MatchMemoryType(device, ~u32(0), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    heap.deviceMemory = AllocateGPUMemory(ctx, logicalDevice, gpuAllocatorSize, deviceMemoryType);

    u32 uniqe[4];
    memcpy(uniqe, &families, sizeof(u32) * 4);
    u32 uniqeCount = get_unique(uniqe, 4);

    auto usageHost = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT   |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferArgs argsHost{uniqe, uniqeCount, usageHost, VK_SHARING_MODE_CONCURRENT, true};
    heap.hostBuffer = MakeVkBuffer(ctx, logicalDevice, heap.uploadAllocator.cap, argsHost);

    auto usageDecie = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT   |
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferArgs argsDev{uniqe, uniqeCount, usageDecie, VK_SHARING_MODE_CONCURRENT, false};
    heap.deviceBuffer = MakeVkBuffer(ctx, logicalDevice, gpuAllocatorSize, argsDev);

    VK_CALL(ctx->vkScratch, vkBindBufferMemory, logicalDevice, heap.hostBuffer, heap.hostMemory, 0);
    VK_CALL(ctx->vkScratch, vkBindBufferMemory, logicalDevice, heap.deviceBuffer, heap.deviceMemory, 0);

    auto VkGetAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(logicalDevice, "vkGetBufferDeviceAddressKHR");
    VkBufferDeviceAddressInfo hostAdddessInfo{};
    hostAdddessInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    hostAdddessInfo.buffer = heap.hostBuffer;
    VkBufferDeviceAddressInfo deviceAdddessInfo{};
    deviceAdddessInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    deviceAdddessInfo.buffer = heap.deviceBuffer;

    heap.hostHeapGpuAddress = VkGetAddressKHR(logicalDevice, &hostAdddessInfo);
    heap.deviceHeapGpuAddress = VkGetAddressKHR(logicalDevice, &deviceAdddessInfo);

    return heap;
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

void AllocateDescriptorSets(VkCoreContext* ctx, VkDevice logicalDevice, VkDescriptorPool pool, u32 count, VkDescriptorSetLayout layout, u32 variableDescriptorCount, VkDescriptorSet* result) {
    
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

    VK_CALL(ctx->vkScratch, vkAllocateDescriptorSets, logicalDevice, &allocInfo, result);
}

void GetDeviceName(VkCoreContext* ctx, VkPhysicalDevice device, char* name) {

    VkPhysicalDeviceProperties deviceProperties{};
    VK_CALL(ctx->vkScratch, vkGetPhysicalDeviceProperties, device, &deviceProperties);
    str_cpy(name, deviceProperties.deviceName);
}

/*
void CreatePipeline(VkCoreContext* core, VkGPUContext* gpu, VkPipelineLayout layout, VkRenderPass renderPass, u32 w, u32 h) {

    auto save = core->vkScratch;
    auto vertSource = (char*)linear_allocator_top(&core->vkScratch);
    u64 vertSourceSize = ReadFile("./vertex.spv", (byte*)vertSource, linear_allocator_free_size(&core->vkScratch));
    if(vertSourceSize == ~u64(0)) {
        // VkLog(&ctx->ioLogBuffer, "s", "File not found");
    }
    linear_allocate(&core->vkScratch, vertSourceSize);

    auto fragSource = (char*)linear_allocator_top(&core->vkScratch);
    u64 fragSourceSize = ReadFile("./frag.spv", (byte*)fragSource, linear_allocator_free_size(&core->vkScratch));
    if(fragSourceSize == ~u64(0)) {
        // VkLog(&ctx->ioLogBuffer, "s", "File not found");
    }
    linear_allocate(&core->vkScratch, fragSourceSize);

    auto vertexModule = CreateShaderModule(core, gpu->logicalDevice, vertSource, vertSourceSize);
    auto fragmentModule = CreateShaderModule(core, gpu->logicalDevice, fragSource, fragSourceSize);

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
    viewport.width = (f32)w;
    viewport.height = (f32)h;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent.width = w;
    scissor.extent.height = h;

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
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.pDepthStencilState = &depthStencilStateInfo;

    VkPipeline pipeline
    VK_CALL(core->vkScratch, vkCreateGraphicsPipelines, gpu->logicalDevice, nullptr, 1, &pipelineInfo, &core->vkAllocator, &ctx->graphicsPipeline);
    VK_CALL(core->vkScratch, vkDestroyShaderModule, gpu->logicalDevice, vertexModule, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyShaderModule, gpu->logicalDevice, fragmentModule, &core->vkAllocator);

    ctx->vkCtx.vkScratch = save;
}
*/

void AddDescriptorUpdate(PendingDescriptorUpdates* pending, VkWriteDescriptorSet write, void* info) {
    auto i = pending->count++;
    pending->write[i] = write;
    memcpy(pending->infos + i, info, sizeof(VkDescriptorBufferInfo));
}
u32 UnRegisterTexture(VkExecutionResources* ctx, u32 handle) {

    ctx->textureSlotTable[handle] = ctx->head;
    ctx->head = handle;
}
u32 RegisterTexture(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* ctx, VkTextureInfo texture, VkSampler sampler) {

    auto slot = ctx->head;
    ctx->head = ctx->textureSlotTable[slot];

    auto updateI = ctx->descriptorUpdates.count++;
    if(updateI >= SIZE_OF_ARRAY(ctx->descriptorUpdates.infos)) {
        VK_CALL(core->vkScratch, vkUpdateDescriptorSets, gpu->logicalDevice, 10, ctx->descriptorUpdates.write, 0, nullptr);
        ctx->descriptorUpdates.count = 0;
    }

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
    
    textureDescriptorWrite->dstSet = ctx->globalTextureDescriptor.set;
    
    textureDescriptorWrite->dstArrayElement = slot;
    textureDescriptorWrite->pImageInfo = imgInfo;
    
    return slot;
}

void BeginCmdState(VkCoreContext* core, CircularAllocator* alloc, CmdState* cmd) {

    cmd->cmdsOnRetrire = circular_allocate(alloc, 512);
    cmd->currentCmd = cmd->cmdsOnRetrire;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CALL(core->vkScratch, vkBeginCommandBuffer, cmd->cmd, &beginInfo);
}
void ExecuteCpuCmds(VkGPUContext* gpu, VkExecutionResources* exeRes, byte* cmdsBegin) {

    auto it = (CpuCommand*)cmdsBegin;
    while(it->op != CMD_EOB) {

        switch(it->op) {
        case CMD_FREE_HOST_ALLOC:
            {
                auto cmdFree = (CommandFreeHostAlloc*)it;
                for(u32 i = 0; i*sizeof(Allocation) < cmdFree->len; i++) {
                    linear_free(&gpu->uploadMemory, cmdFree->allocs[i].ptr, cmdFree->allocs[i].size);
                }
                break;
            }
        case CMD_RELEASE_SEMAPHORE:
            {
                auto cmd = (CommandReleaseSemaphore*)it;
                for(u32 i = 0; i*sizeof(VkSemaphore) < cmd->len; i++) {
                    ReleaseResource(&exeRes->semaphorePool, cmd->semaphores[i]);
                }
                break;
            }
        case CMD_RELEASE_DESCRIPTOR:
            {
                auto cmd = (CommandReleaseDescriptor*)it;

                auto pool = (ResourcePool<byte>*)cmd->descPool;
                auto len = cmd->len - 10;
                for(u32 i = 0; i < len; i += cmd->elemSize) {
                    auto dst = pool->resources + cmd->elemSize * pool->resourceCount++;
                    memcpy(dst, cmd->descriptors + i, cmd->elemSize);
                }
                break;
            }
        }

        auto len = it->len;
        it++;
        it = (CpuCommand*)( (byte*)it + len);
    }
}
bool TryRetireCmdState(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, CmdState* cmdState) {

    auto save = core->vkScratch.top;
    auto status = vkGetFenceStatus(gpu->logicalDevice, cmdState->complete);

    if(status == VK_SUCCESS) {
        ReleaseResource(&res->fencePool, cmdState->complete);
        ReleaseResource(cmdState->cmdSource, cmdState->cmd);
        ExecuteCpuCmds(gpu, res, (byte*)cmdState->cmdsOnRetrire + sizeof(CpuCommand));
    }

    core->vkScratch.top = save;
    return status == VK_SUCCESS;
}
void RetireCmdState(VkCoreContext* ctx, VkGPUContext* gpu, VkExecutionResources* res, CmdState* cmdState) {

    VK_CALL(ctx->vkScratch, vkWaitForFences, gpu->logicalDevice, 1, &cmdState->complete, true, ~u64(0));
    ReleaseResource(&res->fencePool, cmdState->complete);
    ReleaseResource(cmdState->cmdSource, cmdState->cmd);
    ExecuteCpuCmds(gpu, res, (byte*)cmdState->cmdsOnRetrire + sizeof(CpuCommand));
}

void FlushDescriptorUpdates(VkCoreContext* core, VkGPUContext* gpu, PendingDescriptorUpdates* pending) {

    if(pending->count) {
        VK_CALL(core->vkScratch, vkUpdateDescriptorSets, gpu->logicalDevice, pending->count, pending->write, 0, nullptr);
        pending->count = 0;
    }
}
u32 RetireInFlightCmd(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res, u32 count, CmdState* cmds) {

    for(u32 i = 0; i < count; i++) {
        bool retired = TryRetireCmdState(core, gpu, res, cmds + i);
        count -= retired;
        auto cpyIndex = count * retired + i * !retired;
        cmds[i] = cmds[cpyIndex];
        i -= retired;
    }
    return count;
}
CmdState AcquireTransferResources(VkExecutionResources* res) {

    CmdState ret{};
    ret.complete = AcquireResource(&res->fencePool);
    ret.cmd = AcquireResource(&res->transferCmdPool);
    ret.cmdSource = &res->transferCmdPool;

    return ret;
}
CmdState AcquireGraphicsResources(VkExecutionResources* res) {

    CmdState ret{};
    ret.complete = AcquireResource(&res->fencePool);
    ret.cmd = AcquireResource(&res->graphicsCmdPool);
    ret.cmdSource = &res->graphicsCmdPool;

    return ret;
}



void EndCmdState(VkCoreContext* core, CmdState* cmd) {

    auto currentCmd = (CpuCommand*)cmd->currentCmd;
    if(currentCmd->op != CMD_EOB) {

        auto end = (byte*)(currentCmd + 1) + currentCmd->len;
        CpuCommand* endCmd = (CpuCommand*)end;
        endCmd->op = CMD_EOB;
        endCmd->len = 0;

        cmd->currentCmd = endCmd;
    }
    else {
        currentCmd->len = 0;
    }

    VK_CALL(core->vkScratch, vkEndCommandBuffer, cmd->cmd);
}
void IssueGPUCommands(VkCoreContext* ctx, VkGPUContext* gpu, VkQueue queue,  CmdState* cmd, u32 waitCount, VkSemaphore* wait, VkPipelineStageFlags* stages, u32 signalCount, VkSemaphore* signal) {

    VkSubmitInfo sumbitInfo{};
    sumbitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sumbitInfo.commandBufferCount = 1;
    sumbitInfo.pCommandBuffers = &cmd->cmd;
    sumbitInfo.signalSemaphoreCount = signalCount;
    sumbitInfo.waitSemaphoreCount = waitCount;
    sumbitInfo.pWaitSemaphores = wait;
    sumbitInfo.pSignalSemaphores = signal;
    sumbitInfo.pWaitDstStageMask = stages;

    VK_CALL(ctx->vkScratch, vkResetFences, gpu->logicalDevice, 1, &cmd->complete);
    VK_CALL(ctx->vkScratch, vkQueueSubmit, queue, 1, &sumbitInfo, cmd->complete);
}
void IssueFBOPresent(VkGPUContext* gpu, VkFbo* fbo, u32 imgIndex, VkSemaphore wait) {
    
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &wait;
    presentInfo.swapchainCount = 1;
    presentInfo.pImageIndices = &imgIndex;
    presentInfo.pSwapchains = &fbo->swapChain;

    auto res = vkQueuePresentKHR(gpu->presentQueue, &presentInfo);
}


bool AreRenderResourcesReady(VkExecutionResources* res) {

    bool available = IsResourceAvailable(&res->globalRenderParamDescriptors);
    available &= IsResourceAvailable(&res->fencePool);
    available &= IsResourceAvailable(&res->graphicsCmdPool);
    available &= IsResourceAvailable(&res->semaphorePool);
    return available;
}

void RecordDrawDebug(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkProgram* program, CmdState* cmdState, u32 imgIndex, u32 setCount, VkDescriptorSet* descriptors, u32 drawCount, DrawInfo* draws, u64 instanceOffset) {

    VkClearValue clearColor[2]{};
    clearColor[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearColor[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = program->renderPass;
    renderPassInfo.framebuffer = fbo->frames[imgIndex].frameBuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {fbo->width, fbo->height};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearColor;

    VK_CALL(core->vkScratch, vkCmdBeginRenderPass,     cmdState->cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VK_CALL(core->vkScratch, vkCmdBindPipeline, cmdState->cmd, program->bindpoint, program->graphicsPipeline0);
    VkViewport viewport;
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (f32)fbo->width;
    viewport.height = (f32)fbo->height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VK_CALL(core->vkScratch, vkCmdSetViewport, cmdState->cmd, 0, 1, &viewport);
    VkRect2D scissor;
    scissor.offset = {0,0};
    scissor.extent = {fbo->width, fbo->height};
    VK_CALL(core->vkScratch, vkCmdSetScissor, cmdState->cmd, 0, 1, &scissor);

    VK_CALL(core->vkScratch, vkCmdBindDescriptorSets, cmdState->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, program->pipelineLayout, 0, setCount, descriptors, 0, nullptr);

    VkBuffer buffers[2] = {gpu->deviceBuffer, gpu->deviceBuffer};
    u64 offsets[2] = {0, instanceOffset};

    for(u32 k = 0; k < drawCount; k++) {

        offsets[0] = draws[k].model.vertexOffset;
        VK_CALL(core->vkScratch, vkCmdBindVertexBuffers, cmdState->cmd, 0, 2, buffers, offsets);
        VK_CALL(core->vkScratch, vkCmdBindIndexBuffer,   cmdState->cmd, gpu->deviceBuffer, draws[k].model.indexOffset, VK_INDEX_TYPE_UINT32);
        VK_CALL(core->vkScratch, vkCmdDrawIndexed,       cmdState->cmd, draws[k].model.indexCount, draws[k].instanceCount, 0, 0, 0);

        offsets[1] += draws[k].instanceCount * sizeof(InstanceInfo);
    }
    VK_CALL(core->vkScratch, vkCmdEndRenderPass, cmdState->cmd);

}
void RecordGPUDraw(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkProgram2* program, CmdState* cmdState, u32 setCount, VkDescriptorSet* descriptors , u32 imgIndex, u32 drawCount, DrawInfo* draws, u64 instanceOffset) {

    VkClearValue clearColor[2]{};
    clearColor[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearColor[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = program->renderPass;
    renderPassInfo.framebuffer = fbo->frames[imgIndex].frameBuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {fbo->width, fbo->height};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearColor;

    VK_CALL(core->vkScratch, vkCmdBeginRenderPass, cmdState->cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    u32 k = 0;
    for(u32 i = 0; i < program->subpassCount; i++) {

        auto bindPoint = program->subpasses[i].bindpoint;
        for(u32 j = 0; j < program->subpasses[i].pipelineCount; j++) {

            auto pipeline = program->subpasses[i].pipelines[j].pipeline;
            auto layout = program->subpasses[i].pipelines[j].layout;

            VK_CALL(core->vkScratch, vkCmdBindPipeline, cmdState->cmd, bindPoint, pipeline);
            VK_CALL(core->vkScratch, vkCmdBindDescriptorSets, cmdState->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, setCount, descriptors, 0, nullptr);

            VkViewport viewport;
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (f32)fbo->width;
            viewport.height = (f32)fbo->height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VK_CALL(core->vkScratch, vkCmdSetViewport, cmdState->cmd, 0, 1, &viewport);
            VkRect2D scissor;
            scissor.offset = {0,0};
            scissor.extent = {fbo->width, fbo->height};
            VK_CALL(core->vkScratch, vkCmdSetScissor, cmdState->cmd, 0, 1, &scissor);

            VkBuffer buffers[2] = {gpu->deviceBuffer, gpu->deviceBuffer};
            u64 offsets[2] = {0, instanceOffset};


            bool advanceSubpass = false;
            for(; k < drawCount; k++) {

                auto draw = draws++;
                advanceSubpass = draw->advanceSubpass;
                if(draw->advanceSubpass || draw->advancePipeline) {
                    break;
                }

                offsets[0] = draw->model.vertexOffset;
                VK_CALL(core->vkScratch, vkCmdBindIndexBuffer,     cmdState->cmd, gpu->deviceBuffer, draw->model.indexOffset, VK_INDEX_TYPE_UINT32);

                if(draw->instanceCount == 0) {
                    VK_CALL(core->vkScratch, vkCmdBindVertexBuffers,   cmdState->cmd, 0, 1, buffers, offsets);
                    VK_CALL(core->vkScratch, vkCmdDrawIndexed,         cmdState->cmd, draw->model.indexCount, 1, 0, 0, 0);
                }
                else {
                    VK_CALL(core->vkScratch, vkCmdBindVertexBuffers,   cmdState->cmd, 0, 2, buffers, offsets);
                    VK_CALL(core->vkScratch, vkCmdDrawIndexed,         cmdState->cmd, draw->model.indexCount, draw->instanceCount, 0, 0, 0);
                }

                offsets[1] += draw->instanceCount * sizeof(InstanceInfo);
            }

            if(advanceSubpass) {
                break;
            }
        }

        if(i != 0) {
            VK_CALL(core->vkScratch, vkCmdNextSubpass, cmdState->cmd, VK_SUBPASS_CONTENTS_INLINE);
        }
    }
    VK_CALL(core->vkScratch, vkCmdEndRenderPass, cmdState->cmd);
}

u32 IssueSwapChainAcquire(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkSemaphore signalSemaphore, VkFence signalFence) {
    
    u32 imgIndex;

    auto top = core->vkScratch.top;
    VkResult imgAcquireResult = vkAcquireNextImageKHR(gpu->logicalDevice, fbo->swapChain, ~u64(0), signalSemaphore, signalFence, &imgIndex);
    core->vkScratch.top = top;
    
    bool success = (imgAcquireResult == VK_SUBOPTIMAL_KHR) || (imgAcquireResult == VK_ERROR_OUT_OF_DATE_KHR);
    imgIndex = success ? ~u32(0) : imgIndex;

    return imgIndex;
}


void DestroyTexture(VkCoreContext* core, VkGPUContext* gpu, VkTextureInfo texture) {
    
    VK_CALL(core->vkScratch, vkDestroyImageView, gpu->logicalDevice, texture.view, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyImage, gpu->logicalDevice, texture.img, &core->vkAllocator);
}
void RecordGPUCopyBarrier(VkCoreContext* core, VkGPUContext* gpu, CmdState* cmdState, VkAccessFlags dstMask, VkAccessFlags dstStage, MemBlock dst, void* src) {

    VkBufferCopy copy{};
    copy.size = dst.size;
    copy.srcOffset = (byte*)src - gpu->uploadMemory.base;
    copy.dstOffset = dst.offset;
    VK_CALL(core->vkScratch, vkCmdCopyBuffer, cmdState->cmd, gpu->hostBuffer, gpu->deviceBuffer, 1, &copy);

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.buffer = gpu->deviceBuffer;
    barrier.offset = copy.dstOffset;
    barrier.size = copy.size;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = dstMask;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, cmdState->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void MakeVkCoreContext(VkCoreContext* dst, VkRenderContextConfig config, Printer printer, LinearAllocator* alloc) {

    *dst = {};
    dst->ioLog = make_linear_allocator(linear_allocate(alloc, Kilobyte(4)), Kilobyte(4));
    dst->vkScratch = make_linear_allocator(linear_allocate(alloc, config.scractchSize), config.scractchSize);
    
    dst->vkHeap = make_local_malloc((byte*)linear_allocate(alloc, config.vkHeapSize), config.vkHeapSize);
    dst->vkAllocator = {};
    dst->vkAllocator.pUserData       = dst;
    dst->vkAllocator.pfnAllocation   = vkLocalMalloc;
    dst->vkAllocator.pfnFree         = vkLocalFree;
    dst->vkAllocator.pfnReallocation = vkLocalRealloc;
    
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo{};
    if constexpr(ENABLE_VALIDATION_LAYER) {

        LOG_ASSERT(CheckValidationLayerSupport(VALIDATION_LAYERS, SIZE_OF_ARRAY(VALIDATION_LAYERS)), "validation layers not found");

        createInfo.enabledLayerCount = SIZE_OF_ARRAY(VALIDATION_LAYERS);
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS;

        dst->logSeverity = config.logMask;

        debugMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugMessengerCreateInfo.messageSeverity =  config.logMask;
                                                    // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT    |
                                                    // VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                    // VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        debugMessengerCreateInfo.messageType =  VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

        debugMessengerCreateInfo.pfnUserCallback = DebugCallback;
        debugMessengerCreateInfo.pUserData = dst;
        debugMessengerCreateInfo.pNext;
        createInfo.pNext = &debugMessengerCreateInfo;
    }

    {
        auto save = dst->vkScratch.top;
        auto requiredExt = (const char**)linear_allocator_top(&dst->vkScratch);
        memcpy(requiredExt, INSTANCE_EXTENSIONS, sizeof(INSTANCE_EXTENSIONS));
        createInfo.enabledExtensionCount = SIZE_OF_ARRAY(INSTANCE_EXTENSIONS);

        if constexpr (ENABLE_VALIDATION_LAYER) {
            requiredExt[createInfo.enabledExtensionCount++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        }
        createInfo.ppEnabledExtensionNames = requiredExt;
        dst->vkScratch.top += createInfo.enabledExtensionCount * sizeof(const char**);

        VK_CALL(dst->vkScratch, vkCreateInstance, &createInfo, &dst->vkAllocator, &dst->vkInstance);
        dst->vkScratch.top = save;
    }

    if constexpr (ENABLE_VALIDATION_LAYER) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(dst->vkInstance, "vkCreateDebugUtilsMessengerEXT");
        ASSERT(func);
        VK_CALL(dst->vkScratch, func, dst->vkInstance, &debugMessengerCreateInfo, &dst->vkAllocator, &dst->vkDebugMessenger);
    }
}

VkGPUContext MakeVkGPUContext(VkCoreContext* core, VkPhysicalDevice gpu, VkSurfaceKHR surface, VkRenderContextConfig config, Printer printer, LinearAllocator* alloc) {

    VkGPUContext ret{};

    auto mem = linear_allocate(alloc, config.gpuhHeapMaxAllocCount * sizeof(GpuMemoryBlock));
    ret.gpuAllocator = make_gpu_heap(mem, config.gpuhHeapMaxAllocCount, config.gpuHeapSize);

    ret.device = gpu;
    ASSERT(gpu);
    ret.families = GetQueueFamilies(ret.device, surface);
    ret.logicalDevice = CreateLogicalDevice(core, ret.families, ret.device);
    
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.graphicsFamily, 0, &ret.graphicsQueue);
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.presentFamily,  0, &ret.presentQueue);
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.computeFamily,  0, &ret.computeQueue);
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.transferFamily, 0, &ret.transferQueue);

    mem = linear_allocate(alloc, config.uploadHeapSize);
    ret.totalMemorySize = GetGPUMemorySize(gpu);
    auto heap = CreateHeaps(core, ret.device, ret.logicalDevice, ret.families, (byte*)mem, config.uploadHeapSize, config.gpuHeapSize);
    ret.uploadMemory               = heap.uploadAllocator;
    ret.deviceBuffer               = heap.deviceBuffer;
    ret.hostBuffer                 = heap.hostBuffer;
    ret.hostMemoryDeviceAddress    = heap.hostHeapGpuAddress;
    ret.hostMemoryDeviceAddress    = heap.deviceHeapGpuAddress;
    ret.deviceMemory               = heap.deviceMemory;
    ret.hostMemory                 = heap.hostMemory;

    VkCommandPoolCreateInfo graphicsPoolInfo{};
    graphicsPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    graphicsPoolInfo.queueFamilyIndex = ret.families.graphicsFamily;
    graphicsPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPoolCreateInfo transferPoolInfo{};
    transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transferPoolInfo.queueFamilyIndex = ret.families.transferFamily;
    transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CALL(core->vkScratch, vkCreateCommandPool, ret.logicalDevice, &graphicsPoolInfo, &core->vkAllocator, &ret.commandPoolGraphics);
    VK_CALL(core->vkScratch, vkCreateCommandPool, ret.logicalDevice, &transferPoolInfo, &core->vkAllocator, &ret.commandPoolTransfer);

    // 4, 3, 512
    VkDescriptorPoolSize poolSize[2]{};
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = 3;
    poolSize[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize[1].descriptorCount = 512;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSize;
    poolInfo.maxSets = 4;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT;
    VK_CALL(core->vkScratch, vkCreateDescriptorPool, ret.logicalDevice, &poolInfo, &core->vkAllocator, &ret.descriptorPool);

    return ret;
}

VkExecutionResources MakeVkExecutionResources(VkCoreContext* core, VkGPUContext* gpu, Printer printer, LinearAllocator* alloc) {

    VkExecutionResources ret{};
    ret.head = 0;
    ret.textureSlotTable[511] = ~u16(0);
    for(u32 i = 0; i < 511; i++) {
        ret.textureSlotTable[i] = i + 1;
    }

    ret.cmds = (CmdState*)linear_allocate(alloc, sizeof(CmdState) * 6);
    ret.inFlightCmds = 0;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    ret.fencePool.resourceCount = 9;
    ret.fencePool.resources = (VkFence*)linear_allocate(alloc, sizeof(VkFence) * ret.fencePool.resourceCount);

    for(u32 i = 0; i < ret.fencePool.resourceCount; i++) {
        VK_CALL(core->vkScratch, vkCreateFence, gpu->logicalDevice, &fenceInfo, &core->vkAllocator, ret.fencePool.resources + i);
    }

    ret.semaphorePool.resourceCount = 9;
    ret.semaphorePool.resources = (VkSemaphore*)linear_allocate(alloc, sizeof(VkSemaphore) * ret.fencePool.resourceCount);;
    for(u32 i = 0; i < ret.semaphorePool.resourceCount; i++) {
        VK_CALL(core->vkScratch, vkCreateSemaphore, gpu->logicalDevice, &semaphoreInfo, &core->vkAllocator, ret.semaphorePool.resources + i);
    }

    VkCommandBufferAllocateInfo graphicsAllocInfo{};
    graphicsAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    graphicsAllocInfo.commandPool = gpu->commandPoolGraphics;
    graphicsAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    graphicsAllocInfo.commandBufferCount = 2;

    ret.graphicsCmdPool.resourceCount = 2;
    ret.graphicsCmdPool.resources = (VkCommandBuffer*)linear_allocate(alloc, sizeof(VkCommandBuffer) * ret.graphicsCmdPool.resourceCount);
    VK_CALL(core->vkScratch, vkAllocateCommandBuffers, gpu->logicalDevice, &graphicsAllocInfo, ret.graphicsCmdPool.resources);

    VkCommandBufferAllocateInfo transferAllocInfo{};
    transferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    transferAllocInfo.commandPool = gpu->commandPoolTransfer;
    transferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    transferAllocInfo.commandBufferCount = 2;

    ret.transferCmdPool.resourceCount = 2;
    ret.transferCmdPool.resources = (VkCommandBuffer*)linear_allocate(alloc, sizeof(VkCommandBuffer) * ret.transferCmdPool.resourceCount);
    VK_CALL(core->vkScratch, vkAllocateCommandBuffers, gpu->logicalDevice, &transferAllocInfo, ret.transferCmdPool.resources);

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
    VK_CALL(core->vkScratch, vkCreateSampler, gpu->logicalDevice, &samplerInfo, &core->vkAllocator, &ret.textureSampler);

    ret.descriptorUpdates = {};
    ret.cpuCmdAlloc = make_circular_allocator(linear_allocate(alloc, Kilobyte(32)), Kilobyte(32));

    ret.globalRenderParamDescriptors.resourceCount = 3;
    ret.globalRenderParamDescriptors.resources = (DescriptorSet<1>*)linear_allocator_top(alloc);
    ret.layout0 = MakeDescriptorSets(core, gpu, &global_render_info_descriptor_layout_info, 3, alloc);

    auto a = make_linear_allocator(&ret.globalTextureDescriptor, sizeof(ret.globalTextureDescriptor));
    ret.layout1 = MakeDescriptorSets(core, gpu, &global_textures_descriptor_layout_info, 1, &a);

    return ret;
}
void InFlight(VkExecutionResources* exe, CmdState* cmd) {
    exe->cmds[exe->inFlightCmds++] = *cmd;
}

VkFbo MakeVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkRenderPass renderPass, VkSurfaceKHR surface, u32 frameCount, xcb_connection_t* connection, xcb_context* xcb_ctx, Printer printer, LinearAllocator* alloc) {

    VkFbo ret{};

    ret.surface = surface;
    auto res = CreateSwapChain(core, frameCount, gpu->device, gpu->logicalDevice, ret.surface, {xcb_ctx->width, xcb_ctx->height}, gpu->families, nullptr);
    
    ret.width = res.dims.x;
    ret.height = res.dims.y;
    ret.swapChain = res.swapChain;
    ret.fboColorFormat = res.format;
    ret.depthFormat = FindDepthFormat(core, gpu->device);
    u32 families[2] = {gpu->families.graphicsFamily, gpu->families.presentFamily};
    
    u32 imageCount = 0;
    VK_CALL(core->vkScratch, vkGetSwapchainImagesKHR, gpu->logicalDevice, ret.swapChain, &imageCount, nullptr);
    VkImage images[imageCount];
    VK_CALL(core->vkScratch, vkGetSwapchainImagesKHR, gpu->logicalDevice, ret.swapChain, &imageCount, images);

    ret.frameCount = imageCount;
    ret.frames = (SwapChainFrame*)linear_allocate(alloc, ret.frameCount * sizeof(SwapChainFrame));
    
    for(u32 i = 0; i < imageCount; i++) {

        ret.frames[i].colorImg = images[i];
        ret.frames[i].depthImg.img = CreateImg2D(core, gpu->logicalDevice, {ret.width, ret.height}, families, 1, ret.depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        ret.frames[i].depthImg.memory = BackImgMemory(core, gpu->logicalDevice, gpu->deviceMemory, &gpu->gpuAllocator, ret.frames[i].depthImg.img);

        VkImageViewCreateInfo depthCreateInfo{};
        depthCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthCreateInfo.image    = ret.frames[i].depthImg.img;
        depthCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthCreateInfo.format   = ret.depthFormat;

        depthCreateInfo.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthCreateInfo.subresourceRange.baseMipLevel    = 0;
        depthCreateInfo.subresourceRange.levelCount      = 1;
        depthCreateInfo.subresourceRange.baseArrayLayer  = 0;
        depthCreateInfo.subresourceRange.layerCount      = 1;

        VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &depthCreateInfo, &core->vkAllocator, &ret.frames[i].depthImgView);

        VkImageViewCreateInfo colorCreateInfo{};
        colorCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorCreateInfo.image    = ret.frames[i].colorImg;
        colorCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorCreateInfo.format   = ret.fboColorFormat;

        colorCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        colorCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        colorCreateInfo.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
        colorCreateInfo.subresourceRange.baseMipLevel    = 0;
        colorCreateInfo.subresourceRange.levelCount      = 1;
        colorCreateInfo.subresourceRange.baseArrayLayer  = 0;
        colorCreateInfo.subresourceRange.layerCount      = 1;

        VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &colorCreateInfo, &core->vkAllocator, &ret.frames[i].colorImgView);

        VkImageView attachments[] = {ret.frames[i].colorImgView, ret.frames[i].depthImgView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = SIZE_OF_ARRAY(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = ret.width;
        framebufferInfo.height = ret.height;
        framebufferInfo.layers = 1;

        VK_CALL(core->vkScratch , vkCreateFramebuffer, gpu->logicalDevice, &framebufferInfo, &core->vkAllocator, &ret.frames[i].frameBuffer);
    }

    return ret;
}

VkProgram MakeVkProgram(VkCoreContext* core, VkGPUContext* gpu, VkRenderPass renderPass, Printer printer, LinearAllocator* alloc) {

    VkProgram ret{};
    
    /*
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

    ret.UBOdescriptorLayout = CreateDescriptorSetLayouts(core, gpu->logicalDevice, &bufferDescriptorBinding, &nullFlag, 1);
    ret.textureDescriptorLayout = CreateDescriptorSetLayouts(core, gpu->logicalDevice, &textureDescriptorBinding, &bindlessFlags, 1);

    VkDescriptorSet sets[3];
    AllocateDescriptorSets(core, gpu->logicalDevice, gpu->descriptorPool, 1, ret.textureDescriptorLayout, 512, &ret.textureDescriptors);

    AllocateDescriptorSets(core, gpu->logicalDevice, gpu->descriptorPool, 3, ret.UBOdescriptorLayout, 0, sets);
    VkWriteDescriptorSet descriptorWrites[3];
    VkDescriptorBufferInfo bufferInfo[3];

    ret.descriptorSetPool.resourceCount = 3;
    ret.descriptorSetPool.resources = (DescriptorSet<1>*)linear_allocate(alloc, sizeof(DescriptorSet<1>) * ret.descriptorSetPool.resourceCount);

    for(u32 i = 0; i < ret.descriptorSetPool.resourceCount; i++) {
        
        auto block = allocate_gpu_block(&gpu->gpuAllocator, sizeof(GlobalRenderParams), 256);
        ret.descriptorSetPool.resources[i].descriptorMemBlocks[0] = block;
        ret.descriptorSetPool.resources[i].set = sets[i];

        bufferInfo[i] = {};
        bufferInfo[i].buffer = gpu->deviceBuffer;
        bufferInfo[i].offset = block.offset;
        bufferInfo[i].range = block.size;

        descriptorWrites[i] = {};
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].dstBinding = 0;
        descriptorWrites[i].dstSet = sets[i];
        descriptorWrites[i].pBufferInfo = bufferInfo + i;
    }
    VK_CALL(core->vkScratch, vkUpdateDescriptorSets, gpu->logicalDevice, 3, descriptorWrites, 0, nullptr);
    */

    VkDescriptorSetLayout layouts[2]{ret.UBOdescriptorLayout, ret.textureDescriptorLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = layouts;
    VK_CALL(core->vkScratch, vkCreatePipelineLayout, gpu->logicalDevice, &pipelineLayoutInfo, &core->vkAllocator, &ret.pipelineLayout);

    ret.renderPass = renderPass;
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
    auto vertfile = "./vertex3d.spv";
    auto fragfile = "./textured.spv";

    ret.graphicsPipeline0 = CreateGraphicsPipeline(core, gpu->logicalDevice, ret.pipelineLayout, ret.renderPass, &vertexInputInfo, vertfile, fragfile, 640, 480);

    return ret;
}

void DestroyVkCore(VkCoreContext* core) {
   
    if constexpr(ENABLE_VALIDATION_LAYER) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(core->vkInstance, "vkDestroyDebugUtilsMessengerEXT");
        ASSERT(func);
        VK_CALL(core->vkScratch, func, core->vkInstance, core->vkDebugMessenger, &core->vkAllocator);
    }
    VK_CALL(core->vkScratch, vkDestroyInstance, core->vkInstance, &core->vkAllocator);
}
void DestroyVkGPU(VkCoreContext* core, VkGPUContext* gpu) {

    VK_CALL(core->vkScratch, vkDestroyBuffer, gpu->logicalDevice,   gpu->hostBuffer,    &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyBuffer, gpu->logicalDevice,   gpu->deviceBuffer,  &core->vkAllocator);
    VK_CALL(core->vkScratch, vkFreeMemory, gpu->logicalDevice,      gpu->hostMemory,    &core->vkAllocator);
    VK_CALL(core->vkScratch, vkFreeMemory, gpu->logicalDevice,      gpu->deviceMemory,  &core->vkAllocator);

    VK_CALL(core->vkScratch, vkDestroyDescriptorPool, gpu->logicalDevice, gpu->descriptorPool, &core->vkAllocator);

    VK_CALL(core->vkScratch, vkDestroyCommandPool, gpu->logicalDevice, gpu->commandPoolGraphics, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyCommandPool, gpu->logicalDevice, gpu->commandPoolTransfer, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyDevice, gpu->logicalDevice, &core->vkAllocator);
}
void DestroyVkExecutionResources(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* res) {

    VK_CALL(core->vkScratch, vkDeviceWaitIdle, gpu->logicalDevice);

    VK_CALL(core->vkScratch, vkDestroySampler, gpu->logicalDevice, res->textureSampler, &core->vkAllocator);
    free_gpu_block(&gpu->gpuAllocator, res->globalTextureDescriptor.descriptorMemBlocks[0]);
    VK_CALL(core->vkScratch, vkDestroyDescriptorSetLayout, gpu->logicalDevice, res->layout0, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyDescriptorSetLayout, gpu->logicalDevice, res->layout1, &core->vkAllocator);

    for(u32 i = 0; i < res->semaphorePool.resourceCount; i++) {
        VK_CALL(core->vkScratch, vkDestroySemaphore, gpu->logicalDevice, res->semaphorePool.resources[i], &core->vkAllocator);
    }
    for(u32 i = 0; i < res->fencePool.resourceCount; i++) {
        VK_CALL(core->vkScratch, vkDestroyFence, gpu->logicalDevice, res->fencePool.resources[i], &core->vkAllocator);
    }
    VK_CALL(core->vkScratch, vkFreeCommandBuffers, gpu->logicalDevice, gpu->commandPoolTransfer, res->transferCmdPool.resourceCount, res->transferCmdPool.resources);
    VK_CALL(core->vkScratch, vkFreeCommandBuffers, gpu->logicalDevice, gpu->commandPoolGraphics, res->graphicsCmdPool.resourceCount, res->graphicsCmdPool.resources);

}
void DestroyVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo) {

    for(u32 i = 0; i < fbo->frameCount; i++) {

        VK_CALL(core->vkScratch, vkDestroyImageView,    gpu->logicalDevice, fbo->frames[i].depthImgView, &core->vkAllocator);
        VK_CALL(core->vkScratch, vkDestroyImageView,    gpu->logicalDevice, fbo->frames[i].colorImgView, &core->vkAllocator);

        VK_CALL(core->vkScratch, vkDestroyImage,        gpu->logicalDevice, fbo->frames[i].depthImg.img, &core->vkAllocator);
        VK_CALL(core->vkScratch, vkDestroyFramebuffer,  gpu->logicalDevice, fbo->frames[i].frameBuffer, &core->vkAllocator);
    }

    VK_CALL(core->vkScratch, vkDestroySwapchainKHR, gpu->logicalDevice, fbo->swapChain, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroySurfaceKHR, core->vkInstance, fbo->surface, &core->vkAllocator);
}

PipelineInfo MakePipeline(VkCoreContext* core, VkGPUContext* gpu, u32 subpassIndex, VkRenderPass renderPass, u32 descLayoutCount, VkDescriptorSetLayout* descLayouts, const PipelineDescriptor* pipeDescription) {

    PipelineInfo ret{};

    VkPipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = descLayoutCount;
    layoutCreateInfo.pSetLayouts = descLayouts;
    VkPipelineLayout layout;
    VK_CALL(core->vkScratch, vkCreatePipelineLayout, gpu->logicalDevice, &layoutCreateInfo, &core->vkAllocator, &layout);
    ret.layout = layout;
    
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

    auto fragCode = pipeDescription->fragByteCode;
    auto vertCode = pipeDescription->vertByteCode;
    stages[0].module = CreateShaderModule(core, gpu->logicalDevice, vertCode.mem, vertCode.size);
    stages[1].module = CreateShaderModule(core, gpu->logicalDevice, fragCode.mem, fragCode.size);

    stages[0].pName = "main";
    stages[1].pName = "main";

    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = stages;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.pVertexAttributeDescriptions    = pipeDescription->attribDescriptor.attribs;
    vertexInput.pVertexBindingDescriptions      = pipeDescription->attribDescriptor.bindings;
    vertexInput.vertexAttributeDescriptionCount = pipeDescription->attribDescriptor.attribCount;
    vertexInput.vertexBindingDescriptionCount   = pipeDescription->attribDescriptor.bindingCount;

    pipelineCreateInfo.pVertexInputState   = &vertexInput;
    pipelineCreateInfo.pInputAssemblyState = &pipeDescription->inputAsm;
    pipelineCreateInfo.pTessellationState  = &pipeDescription->tessellationState;
    pipelineCreateInfo.pViewportState      = &pipeDescription->viewportState;
    pipelineCreateInfo.pRasterizationState = &pipeDescription->rasterizationState;
    pipelineCreateInfo.pMultisampleState   = &pipeDescription->multisampleState;
    pipelineCreateInfo.pDepthStencilState  = &pipeDescription->depthStencilState;
    pipelineCreateInfo.pColorBlendState    = &pipeDescription->colorBlendState;
    pipelineCreateInfo.pDynamicState       = &pipeDescription->dynamicState;

    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.layout = layout;
    pipelineCreateInfo.subpass = subpassIndex;

    VkPipeline pipeline;
    VK_CALL(core->vkScratch, vkCreateGraphicsPipelines, gpu->logicalDevice,  nullptr, 1, &pipelineCreateInfo, &core->vkAllocator, &pipeline);
    ret.pipeline = pipeline;

    VK_CALL(core->vkScratch, vkDestroyShaderModule, gpu->logicalDevice, stages[0].module, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyShaderModule, gpu->logicalDevice, stages[1].module, &core->vkAllocator);

    return ret;
}

struct DescriptorSetRuntime {
    VkDescriptorSet set;
    MemBlock descriptorMemBlocks[];
};
u32 ComputeDescriptorSize(const DescriptorSetLayoutInfo* info) {

    DescriptorSetRuntime* d = nullptr;
    return (u64)(d->descriptorMemBlocks + info->bindingCount);
}
VkDescriptorSetLayout MakeDescriptorSets(VkCoreContext* core, VkGPUContext* gpu, const DescriptorSetLayoutInfo* info, u32 setCount, LinearAllocator* alloc) {

    VkDescriptorSetLayout layout;
    auto descriptorCount = info->bindingCount;
    auto flags           = info->flags;
    auto layoutBindings  = info->layoutBindings;
    layout = CreateDescriptorSetLayouts(core, gpu->logicalDevice, layoutBindings, flags, descriptorCount);

    u32 varCount = 0;
    if(info->flags[descriptorCount - 1] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT) {
        varCount = info->layoutBindings[descriptorCount - 1].descriptorCount;
    }
    VkDescriptorSet sets[setCount];
    AllocateDescriptorSets(core, gpu->logicalDevice, gpu->descriptorPool, setCount, layout, varCount, sets);

    VkWriteDescriptorSet descriptorWrites[setCount * descriptorCount]{};
    VkDescriptorBufferInfo bufferInfo[setCount * descriptorCount]{};
    u32 writeCount = 0;

    auto size = ComputeDescriptorSize(info);
    for(u32 i = 0; i < setCount; i++) {

        auto set = (DescriptorSetRuntime*)linear_allocate(alloc, size);
        set->set = sets[i];

        for(u32 k = 0; k < descriptorCount; k++) {

            auto write = descriptorWrites + writeCount;
            write->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write->dstSet = sets[i];

            auto memInfo = bufferInfo + writeCount;
            write->pBufferInfo = memInfo;
            memInfo->buffer = gpu->deviceBuffer;

            write->descriptorType = info->layoutBindings[k].descriptorType;
            write->dstBinding = info->layoutBindings[k].binding;
            write->dstArrayElement = 0;
            write->descriptorCount = info->layoutBindings[k].descriptorCount;

            writeCount++;
            switch(info->layoutBindings[k].descriptorType) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                set->descriptorMemBlocks[k] = allocate_gpu_block(&gpu->gpuAllocator, info->bindingMemorySizes[k], 256);
                break;
            default:
                writeCount--;
                break;
            }

            memInfo->offset = set->descriptorMemBlocks[k].offset;
            memInfo->range = set->descriptorMemBlocks[k].size;
        }
    }

    VK_CALL(core->vkScratch, vkUpdateDescriptorSets, gpu->logicalDevice, writeCount, descriptorWrites, 0, nullptr);
    return layout;
}


void PushAttribute(VertexDescriptor* dst, u32 location, u32 binding, VkFormat format, u32 offset,  LinearAllocator* alloc) {

    auto desc = (VkVertexInputAttributeDescription*)linear_allocate(alloc, sizeof(VkVertexInputAttributeDescription));
    desc->location = location;
    desc->binding  = binding;
    desc->format   = format;
    desc->offset   = offset;

    if(dst->attribCount == 0) {
        dst->attribs = desc;
    }
    dst->attribCount++;
}
void PushAttributeBinding(VertexDescriptor* dst, u32 binding, u32 stride, VkVertexInputRate inputRate, LinearAllocator* alloc) {

    auto desc = (VkVertexInputBindingDescription*)linear_allocate(alloc, sizeof(VkVertexInputBindingDescription));
    desc->binding   = binding;
    desc->stride    = stride;
    desc->inputRate = inputRate;

    if(dst->bindingCount == 0) {
        dst->bindings = desc;
    }
    dst->bindingCount++;
}

void PushCPUCommandFreeHost(CmdState* state, void* mem, u32 size) {

    CommandFreeHostAlloc* cmdFree;
    if(*((CpuCMDOp*)state->currentCmd) == CMD_FREE_HOST_ALLOC) {
        cmdFree = (CommandFreeHostAlloc*)state->currentCmd;
    }
    else {
        auto cmd = (CpuCommand*)state->currentCmd;
        auto end = (byte*)(cmd + 1) + cmd->len;
        cmdFree = (CommandFreeHostAlloc*)end;
        cmdFree->op = CMD_FREE_HOST_ALLOC;
        cmdFree->len = 0;
        state->currentCmd = cmdFree;
    }
    auto alloc = (Allocation*)((byte*)cmdFree->allocs + cmdFree->len);
    alloc->ptr = mem;
    alloc->size = size;
    cmdFree->len += sizeof(Allocation);
}
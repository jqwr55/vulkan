#define VULKAN_DEBUG 1
#include <vulkan.h>
#include <debug.h>
#include <common.h>

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
    VK_KHR_DEVICE_GROUP_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_MAINTENANCE3_EXTENSION_NAME,
};
const char* INSTANCE_EXTENSIONS[] = {
    VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME,
};
constexpr u32 MAX_FRAMES_IN_FLIGHT = 1;

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

    auto ctx = (VkCoreContext*)user;
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
    auto ctx = (VkCoreContext*)user;
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
    auto ctx = (VkCoreContext*)user;
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
u32 GetRequiredExtensions(const char** dst, const char** instanceExtensions, u32 instanceExtensionsCount) {

    ASSERT(dst && instanceExtensions);
    const auto begin = dst;
    *dst++ = VK_KHR_SURFACE_EXTENSION_NAME;
    *dst++ = VK_KHR_XCB_SURFACE_EXTENSION_NAME;
    memcpy(dst, instanceExtensions, sizeof(const char*) * instanceExtensionsCount);
    dst += instanceExtensionsCount;

    if constexpr (ENABLE_VALIDATION_LAYER) {
        *dst++ = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    return dst - begin;
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
        VkLog(&ctx->ioLog, "sscsc", "[vulkan validation layer info] severity ", SEVERITY_STR[(i32)f32_log(messageSeverity, 16)], ' ', pCallbackData->pMessage, '\n');
        VkFlushLog(&ctx->ioLog);
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

QueueFamilies GetQueueFamilies(VkCoreContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface) {

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
bool IsDeviceSuitable(VkCoreContext* ctx, VkPhysicalDevice device, VkSurfaceKHR surface, const char** requiredDeviceExt, u32 count) {

    auto family = GetQueueFamilies(ctx, device, surface);
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

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos;
    createInfo.queueCreateInfoCount = uniqueFamilyCount;
    createInfo.pNext = &deviceFeatures2;;
    
    createInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS;
    createInfo.enabledExtensionCount = SIZE_OF_ARRAY(DEVICE_EXTENSIONS);

    if constexpr (ENABLE_VALIDATION_LAYER) {
        createInfo.enabledLayerCount = SIZE_OF_ARRAY(VALIDATION_LAYERS);
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS;
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
    swapChainCreateInfo.minImageCount = Clamp(frameCount, capabilities.minImageCount, capabilities.maxImageCount);
    swapChainCreateInfo.preTransform = capabilities.currentTransform;
    VK_CALL(ctx->vkScratch, vkCreateSwapchainKHR, logicalDevice, &swapChainCreateInfo, &ctx->vkAllocator, &ret.swapChain);

    ASSERT(frameCount > capabilities.minImageCount && frameCount < capabilities.maxImageCount);
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

void RecordVkTextureUpload(VkCoreContext* core, VkGPUContext* gpu, CmdState* transfer, CmdState* graphics, VkTextureInfo vkTex, ImageDescriptor img) {

    u32 imgSize = img.width * img.height * 4;
    auto texels = img.img;

    auto dst = (Pixel*)linear_alloc(&gpu->uploadMemory, imgSize);
    memcpy(dst, texels, imgSize);

    CmdFreeHostAlloc* cmdFree;
    if(*((CpuCMDOp*)transfer->currentCmd) == CMD_FREE_HOST_ALLOC) {
        cmdFree = (CmdFreeHostAlloc*)transfer->currentCmd;
    }
    else {
        auto cmd = (CpuCmd*)transfer->currentCmd;
        auto end = (byte*)(cmd + 1) + cmd->len;
        cmdFree = (CmdFreeHostAlloc*)end;
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

VkTextureInfo CreateVkTexture(VkCoreContext* core, VkGPUContext* gpu, ImageDescriptor img) {

    VkTextureInfo ret;
    ret.img = CreateImg2D(core, gpu->logicalDevice, {img.width,img.height}, 0, 0, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ret.memory = BackImgMemory(core, gpu->logicalDevice, gpu->deviceMemory, &gpu->gpuAllocator, ret.img);

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
    VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &viewInfo, &core->vkAllocator, &ret.view);

    return ret;
}

VkTextureInfo UploadVkTexture(VkCoreContext* core, VkGPUContext* gpu, VkExecutionResources* exeRes, ImageDescriptor img) {

    u32 imgSize = img.width * img.height * 4;
    auto texels = img.img;

    auto dst = (Pixel*)linear_top(&gpu->uploadMemory);
    memcpy(dst, texels, imgSize);

    VkTextureInfo ret;
    ret.img = CreateImg2D(core, gpu->logicalDevice, {img.width,img.height}, &gpu->families.graphicsFamily, 1, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ret.memory = BackImgMemory(core, gpu->logicalDevice, gpu->deviceMemory, &gpu->gpuAllocator, ret.img);

    VkCommandBuffer cmd;
    ASSERT(TryAcquireResource(&exeRes->transferCmdPool, &cmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CALL(core->vkScratch, vkBeginCommandBuffer, cmd, &begin);

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
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, cmd, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &toTransferDstOptimal);

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
    VK_CALL(core->vkScratch, vkCmdCopyBufferToImage, cmd, gpu->hostBuffer, ret.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VK_CALL(core->vkScratch, vkEndCommandBuffer, cmd);

    VkCommandBuffer ownership;
    ASSERT(TryAcquireResource(&exeRes->graphicsCmdPool, &ownership));
    VK_CALL(core->vkScratch, vkBeginCommandBuffer, ownership, &begin);

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
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, ownership, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &toShaderReadOPt);
    VK_CALL(core->vkScratch, vkEndCommandBuffer, ownership);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence;
    ASSERT(TryAcquireResource(&exeRes->fencePool, &fence));
    VK_CALL(core->vkScratch, vkResetFences, gpu->logicalDevice, 1, &fence);
    VK_CALL(core->vkScratch, vkQueueSubmit, gpu->transferQueue, 1, &submitInfo, fence);

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
    VK_CALL(core->vkScratch, vkCreateImageView, gpu->logicalDevice, &viewInfo, &core->vkAllocator, &ret.view);

    VK_CALL(core->vkScratch, vkWaitForFences, gpu->logicalDevice, 1, &fence, true, ~u64(0));

    VkSubmitInfo submitInfo2{};
    submitInfo2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo2.commandBufferCount = 1;
    submitInfo2.pCommandBuffers = &ownership;
    VK_CALL(core->vkScratch, vkResetFences, gpu->logicalDevice, 1, &fence);
    VK_CALL(core->vkScratch, vkQueueSubmit, gpu->graphicsQueue, 1, &submitInfo2, fence);
    VK_CALL(core->vkScratch, vkWaitForFences, gpu->logicalDevice, 1, &fence, true, ~u64(0));

    ReleaseResource(&exeRes->fencePool, fence);
    ReleaseResource(&exeRes->transferCmdPool, cmd);
    ReleaseResource(&exeRes->graphicsCmdPool, ownership);

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

VkShaderModule CreateShaderModule(VkCoreContext* ctx, VkDevice logicalDevice, const char* source, u32 len) {

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

VkPipelineLayout CreatePipelineLayout(VkContext* ctx, VkDescriptorSetLayout* descriptorSetLayout, u32 count) {

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = count;
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayout;

    VkPipelineLayout ret;
    VK_CALL(ctx->vkScratch, vkCreatePipelineLayout, ctx->logicalDevice, &pipelineLayoutInfo, &ctx->vkAllocator, &ret);

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
    VK_CALL(ctx->vkScratch, vkCreateRenderPass, logicalDevice, &renderPassInfo, &ctx->vkAllocator, &renderPass);

    return renderPass;
}

VkPipeline CreateGraphicsPipeline(VkCoreContext* ctx, VkDevice logicalDevice, VkPipelineLayout layout, VkRenderPass renderPass, u32 width, u32 height) {

    auto allocSave = ctx->vkScratch;
    auto vertSource = (char*)linear_allocator_top(&ctx->vkScratch);
    u64 vertSourceSize = ReadFile("./vertex.spv", (byte*)vertSource, linear_allocator_free_size(&ctx->vkScratch));
    if(vertSourceSize == ~u64(0)) {
        ASSERT(false);
        // VkLog(&ctx->ioLogBuffer, "s", "vertex.spv not found\n");
        // VkFlushLog(&ctx->ioLogBuffer);
    }
    linear_allocate(&ctx->vkScratch, vertSourceSize);

    auto fragSource = (char*)linear_allocator_top(&ctx->vkScratch);
    u64 fragSourceSize = ReadFile("./frag.spv", (byte*)fragSource, linear_allocator_free_size(&ctx->vkScratch));
    if(fragSourceSize == ~u64(0)) {
        ASSERT(false);
        // VkLog(&ctx->ioLogBuffer, "s", "frag.spv not found\n");
        // VkFlushLog(&ctx->ioLogBuffer);
    }
    linear_allocate(&ctx->vkScratch, fragSourceSize);

    auto vertexModule = CreateShaderModule(ctx, logicalDevice, vertSource, vertSourceSize);
    auto fragmentModule = CreateShaderModule(ctx, logicalDevice, fragSource, fragSourceSize);

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
    pipelineInfo.pVertexInputState = &vertexInputInfo;
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

void CreateFramebuffers(VkRenderContext* ctx, VkRenderPass renderPass, SwapChainFrame* dst, u32 frameCount) {
    
    for (size_t i = 0; i < frameCount; i++) {

        VkImageView attachments[] = {dst[i].colorImgView, dst[i].depthImgView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = SIZE_OF_ARRAY(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = ctx->width;
        framebufferInfo.height = ctx->height;
        framebufferInfo.layers = 1;

        VK_CALL(ctx->vkCtx.vkScratch , vkCreateFramebuffer, ctx->vkCtx.logicalDevice, &framebufferInfo, &ctx->vkCtx.vkAllocator, &dst[i].frameBuffer);
    }
}
void CreateCommandPool(VkRenderContext* ctx) {

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
void CreateCommandBuffers(VkRenderContext* ctx, u32 graphicsCmdCount, u32 transferCmdCount) {
    
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
u32 MatchMemoryType(VkCoreContext* ctx, VkPhysicalDevice device, u32 typeFilter, VkMemoryPropertyFlags properties) {
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);

    for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {
        if( ((typeFilter & (1 << i)) == (1 << i)) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
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
void CreateSyncObjects(VkRenderContext* ctx, u32 count) {

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

    VkDeviceMemory deviceMemory;
    VkDeviceMemory hostMemory;
    VkBuffer deviceBuffer;
    VkBuffer hostBuffer;


    u64 deviceHeapGpuAddress;
    u64 hostHeapGpuAddress;
};

VkHeaps CreateHeaps(VkCoreContext* ctx, VkPhysicalDevice device, VkDevice logicalDevice, QueueFamilies families, byte* localMemory, u32 localMemorySize, u32 gpuAllocatorSize) {

    VkHeaps heap{};

    auto aligment = GetImportMemoryAlignment(ctx, device);
    auto aligned = (byte*)align_pointer(localMemory, aligment);
    localMemorySize -= aligned - localMemory;

    heap.uploadAllocator = make_coalescing_linear_allocator(aligned, localMemorySize);

    auto vkGetMemoryHostPointerPropertiesEXT_ = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(logicalDevice, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT prop{};
    prop.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    VK_CALL(ctx->vkScratch, vkGetMemoryHostPointerPropertiesEXT_, logicalDevice, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, heap.uploadAllocator.base, &prop);
    
    auto importMemoryType = MatchMemoryType(ctx, device, prop.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    heap.hostMemory = ImportMemory(ctx, logicalDevice, heap.uploadAllocator.base, heap.uploadAllocator.cap, importMemoryType);

    auto deviceMemoryType = MatchMemoryType(ctx, device, ~u32(0), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
    heap.hostBuffer = MakeVkBuffer(ctx, logicalDevice, heap.uploadAllocator.cap-256, argsHost);

    auto usageDecie = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT   |
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferArgs argsDev{uniqe, uniqeCount, usageDecie, VK_SHARING_MODE_CONCURRENT, false};
    heap.deviceBuffer = MakeVkBuffer(ctx, logicalDevice, gpuAllocatorSize, argsDev);

    VkMemoryRequirements pMemoryRequirements{};
    VK_CALL(ctx->vkScratch, vkGetBufferMemoryRequirements, logicalDevice, heap.hostBuffer, &pMemoryRequirements);

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

struct DescriptorSetsAllocateInfo {
    VkDescriptorSetLayout layout;
    VkDescriptorSet* result;
    u32 count;
};
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

void UnRegisterTexture(VkRenderContext* ctx, u32 slot) {

    ctx->textureSlotTable[slot] = ctx->head;
    ctx->head = slot;
}
u32 RegisterTexture(VkExecutionResources* ctx, VkTextureInfo texture, VkSampler sampler, VkDescriptorSet set) {

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

/*
u32 MakeVkRenderContext(VkRenderContext* ctx, xcb_connection_t* connection, xcb_context xcb, byte* memory, u32 memorySize, VkRenderContextConfig config) {

    auto requiredSize = config.scractchSize     +
                        config.vkHeapSize       +
                        config.ioLogBufferSize  +
                        config.uploadHeapSize   +
                        config.localHeapSize    +
                        config.gpuHeapSize;
    if(requiredSize > memorySize) {
        return ~u32(0);
    }
    const auto memBase = memory;
    memset(memBase, 0, requiredSize);

    *ctx = {};
    ctx->ioLogBuffer = make_linear_allocator(memory, config.ioLogBufferSize);
    memory += config.ioLogBufferSize;

    ctx->vkCtx.vkScratch = make_linear_allocator(memory, config.scractchSize);
    memory += config.scractchSize;

    ctx->vkCtx.vkHeap = make_local_malloc(memory, config.vkHeapSize);
    memory += config.vkHeapSize;

    ctx->gpuAllocator = make_gpu_heap(memory, config.gpuhHeapMaxAllocCount, config.gpuHeapSize);
    memory += config.gpuhHeapMaxAllocCount * sizeof(GpuMemoryBlock);

    ctx->localHeap = make_local_malloc(memory, config.localHeapSize);
    memory += config.localHeapSize;

    ctx->head = 0;
    ctx->textureSlotTable[511] = ~u16(0);
    for(u32 i = 0; i < 511; i++) {
        ctx->textureSlotTable[i] = i + 1;
    }

    ctx->width = config.windowWidth;
    ctx->height = config.windowHeight;

    ctx->title = "vk_test";
    // create window, set icon
    VkLog(&ctx->ioLogBuffer, "sssicic", "[xcb info] window \"", ctx->title, "\" created ", ctx->width, ' ', ctx->height, '\n');

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

        LOG_ASSERT(CheckValidationLayerSupport(VALIDATION_LAYERS, SIZE_OF_ARRAY(VALIDATION_LAYERS)), "validation layers not found");
        VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] validation layers supported\n");

        createInfo.enabledLayerCount = SIZE_OF_ARRAY(VALIDATION_LAYERS);
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS;

        ctx->logSeverity = config.logMask;
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
    createInfo.enabledExtensionCount = GetRequiredExtensions(requiredExt, INSTANCE_EXTENSIONS, SIZE_OF_ARRAY(INSTANCE_EXTENSIONS));
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

    VkXcbSurfaceCreateInfoKHR xcbSurfaceCreateInfo{};
    xcbSurfaceCreateInfo.connection = connection;
    xcbSurfaceCreateInfo.window = xcb.window;
    xcbSurfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
    VK_CALL(ctx->vkCtx.vkScratch, vkCreateXcbSurfaceKHR, ctx->vkCtx.vkInstance, &xcbSurfaceCreateInfo, &ctx->vkCtx.vkAllocator, &ctx->surface);

    ctx->vkCtx.device = PickPhysicalDevice(&ctx->vkCtx, ctx->vkCtx.vkInstance, ctx->surface, DEVICE_EXTENSIONS, SIZE_OF_ARRAY(DEVICE_EXTENSIONS));
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

    auto heap = CreateHeaps(ctx, memory, config.uploadHeapSize, config.gpuHeapSize);
    memory += config.uploadHeapSize;
    ctx->uploadMemory               = heap.uploadAllocator;
    ctx->deviceBuffer               = heap.deviceBuffer;
    ctx->hostBuffer                 = heap.hostBuffer;
    ctx->hostMemoryDeviceAddress    = heap.hostHeapGpuAddress;
    ctx->hostMemoryDeviceAddress    = heap.deviceHeapGpuAddress;
    ctx->deviceMemory               = heap.deviceMemory;
    ctx->hostMemory                 = heap.hostMemory;
    VkLog(&ctx->ioLogBuffer, "s", "[vulkan info] created host and device memory heaps\n");

    auto chain = CreateSwapChain(&ctx->vkCtx, {ctx->width, ctx->height}, config.fboCount, ctx->vkCtx.device, ctx->vkCtx.logicalDevice, ctx->surface, ctx->families, nullptr);
    ctx->swapChain = chain.swapChain;
    ctx->swapChainImageFormat = chain.format;

    ctx->swapChainFrames.Init(&ctx->localHeap, config.fboCount);
    auto depthFormat = FindDepthFormat(&ctx->vkCtx);
    ctx->swapChainFrames.size = CreateSwapChainFrames(ctx, ctx->swapChainFrames.mem, chain.swapChain, depthFormat, chain.format);
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
    VkDescriptorBufferInfo bufferInfo[3];
    ctx->descriptorSetPool.resourceCount = 3;
    ctx->descriptorSetPool.resources = (Descriptor*)local_malloc(&ctx->vkCtx.vkHeap, sizeof(Descriptor) * ctx->descriptorSetPool.resourceCount);
    for(u32 i = 0; i < ctx->descriptorSetPool.resourceCount; i++) {
        auto block = allocate_gpu_block(&ctx->gpuAllocator, sizeof(CommonParams), 256);
        ctx->descriptorSetPool.resources[i].offset = block.offset;
        ctx->descriptorSetPool.resources[i].set = sets[i];

        bufferInfo[i] = {};
        bufferInfo[i].buffer = ctx->deviceBuffer;
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
    VK_CALL(ctx->vkCtx.vkScratch, vkUpdateDescriptorSets, ctx->vkCtx.logicalDevice, 3, descriptorWrites, 0, nullptr);

    VkDescriptorSetLayout descriptorLayouts[2] = {ctx->UBOdescriptorLayout, ctx->textureDescriptorLayout};
    ctx->pipelineLayout = CreatePipelineLayout(&ctx->vkCtx, descriptorLayouts, 2);
    ctx->renderPass = CreateRenderPass(&ctx->vkCtx, ctx->swapChainImageFormat);
    CreateFramebuffers(ctx, ctx->renderPass, ctx->swapChainFrames.mem, ctx->swapChainFrames.size);
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

    return memory - memBase;
}
*/

void DestroyRenderContext(VkRenderContext* ctx) {

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

    VkLog(&ctx->ioLogBuffer, "sss", "[xcb info] window \"", ctx->title, "\" destroyed\n");
    VkFlushLog(&ctx->ioLogBuffer);
}
void BeginCmdState(VkCoreContext* core, CircularAllocator* alloc, CmdState* cmd) {

    cmd->cmdsOnRetrire = circular_allocate(alloc, 512);
    cmd->currentCmd = cmd->cmdsOnRetrire;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CALL(core->vkScratch, vkBeginCommandBuffer, cmd->cmd, &beginInfo);
}
void ExecuteCpuCmds(VkGPUContext* gpu, VkProgram* program, VkExecutionResources* exeRes, byte* cmdsBegin) {

    auto it = (CpuCmd*)cmdsBegin;
    while(it->op != CMD_EOB) {

        switch(it->op) {
        case CMD_FREE_HOST_ALLOC:
            {
                auto cmdFree = (CmdFreeHostAlloc*)it;
                for(u32 i = 0; i*sizeof(Allocation) < cmdFree->len; i++) {
                    linear_free(&gpu->uploadMemory, cmdFree->allocs[i].ptr, cmdFree->allocs[i].size);
                }
                break;
            }
        case CMD_RELEASE_SEMAPHORE:
            {
                auto cmd = (CmdReleaseSemaphore*)it;
                for(u32 i = 0; i*sizeof(VkSemaphore) < cmd->len; i++) {
                    ReleaseResource(&exeRes->semaphorePool, cmd->semaphores[i]);
                }
                break;
            }
        case CMD_RELEASE_DESCRIPTOR:
            {
                auto cmd = (CmdReleaseDescriptor*)it;
                for(u32 i = 0; i*sizeof(Descriptor) < cmd->len; i++) {
                    ReleaseResource(&program->descriptorSetPool, cmd->descriptors[i]);
                }
                break;
            }
        }

        auto len = it->len;
        it++;
        it = (CpuCmd*)( (byte*)it + len);
    }
}
bool TryRetireCmdState(VkCoreContext* core, VkGPUContext* gpu, VkProgram* program, VkExecutionResources* res, CmdState* cmdState) {

    auto save = core->vkScratch.top;
    auto status = vkGetFenceStatus(gpu->logicalDevice, cmdState->complete);

    if(status == VK_SUCCESS) {
        ReleaseResource(&res->fencePool, cmdState->complete);
        ReleaseResource(cmdState->cmdSource, cmdState->cmd);
        ExecuteCpuCmds(gpu, program, res, (byte*)cmdState->cmdsOnRetrire + sizeof(CpuCmd));
    }

    core->vkScratch.top = save;
    return status == VK_SUCCESS;
}
void RetireCmdState(VkCoreContext* ctx, VkGPUContext* gpu, VkProgram* program, VkExecutionResources* res, CmdState* cmdState) {

    VK_CALL(ctx->vkScratch, vkWaitForFences, gpu->logicalDevice, 1, &cmdState->complete, true, ~u64(0));
    ReleaseResource(&res->fencePool, cmdState->complete);
    ReleaseResource(cmdState->cmdSource, cmdState->cmd);
    ExecuteCpuCmds(gpu, program, res, (byte*)cmdState->cmdsOnRetrire + sizeof(CpuCmd));
}

void FlushDescriptorUpdates(VkCoreContext* core, VkGPUContext* gpu, PendingDescriptorUpdates* pending) {

    if(pending->count) {
        VK_CALL(core->vkScratch, vkUpdateDescriptorSets, gpu->logicalDevice, pending->count, pending->write, 0, nullptr);
        pending->count = 0;
    }
}
u32 RetireInFlightCmd(VkCoreContext* core, VkGPUContext* gpu, VkProgram* program, VkExecutionResources* res, u32 count, CmdState* cmds) {

    for(u32 i = 0; i < count; i++) {
        bool retired = TryRetireCmdState(core, gpu, program, res, cmds + i);
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

    auto currentCmd = (CpuCmd*)cmd->currentCmd;
    auto end = (byte*)(currentCmd + 1);
    auto endCmd = (CpuCmd*)(end + currentCmd->len);
    endCmd->op = CMD_EOB;
    endCmd->len = 0;

    VK_CALL(core->vkScratch, vkEndCommandBuffer, cmd->cmd);
}
void IssueCmdState(VkCoreContext* ctx, VkGPUContext* gpu, VkQueue queue,  CmdState* cmd, u32 waitCount, VkSemaphore* wait, VkPipelineStageFlags* stages, u32 signalCount, VkSemaphore* signal) {


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
void IssuePresentImg(VkGPUContext* gpu, VkFbo* fbo, u32 imgIndex, VkSemaphore wait) {
    
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &wait;
    presentInfo.swapchainCount = 1;
    presentInfo.pImageIndices = &imgIndex;
    presentInfo.pSwapchains = &fbo->swapChain;

    auto res = vkQueuePresentKHR(gpu->presentQueue, &presentInfo);
}


bool AreTransferResourcesReady(VkRenderContext* ctx) {

    auto available = IsResourceAvailable(&ctx->fencePool);
    available &= IsResourceAvailable(&ctx->transferCmdPool);
    available &= IsResourceAvailable(&ctx->semaphorePool);
    return available;
}
bool AreRenderResourcesReady(VkProgram* program, VkExecutionResources* res) {

    auto available = IsResourceAvailable(&program->descriptorSetPool);
    available &= IsResourceAvailable(&res->fencePool);
    available &= IsResourceAvailable(&res->graphicsCmdPool);
    available &= IsResourceAvailable(&res->semaphorePool);
    return available;
}

void RecordDraw(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkProgram* program, CmdState* cmdState, VkDescriptorSet descriptors, u32 imgIndex, u32 drawCount, DrawInfo* draws, u64 instanceOffset) {

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
    VK_CALL(core->vkScratch, vkCmdBindPipeline,        cmdState->cmd, program->bindpoint, program->graphicsPipeline);
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

    VkDescriptorSet boundSets[2] = {descriptors, program->textureDescriptors};
    VK_CALL(core->vkScratch, vkCmdBindDescriptorSets,  cmdState->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, program->pipelineLayout, 0, 2, boundSets, 0, nullptr);

    VkBuffer buffers[2] = {gpu->deviceBuffer, gpu->deviceBuffer};
    u64 offsets[2] = {0, instanceOffset};

    for(u32 i = 0; i < drawCount; i++) {
        offsets[0] = draws[i].model.vertexOffset;

        VK_CALL(core->vkScratch, vkCmdBindVertexBuffers,   cmdState->cmd, 0, 2, buffers, offsets);
        VK_CALL(core->vkScratch, vkCmdBindIndexBuffer,     cmdState->cmd, gpu->deviceBuffer, draws[i].model.indexOffset, VK_INDEX_TYPE_UINT32);
        VK_CALL(core->vkScratch, vkCmdDrawIndexed,         cmdState->cmd, draws[i].model.indexCount, draws[i].instanceCount, 0, 0, 0);

        offsets[1] += draws[i].instanceCount * sizeof(InstanceInfo);
    }

    VK_CALL(core->vkScratch, vkCmdEndRenderPass, cmdState->cmd);
}

u32 IssueSwapChainAcquire(VkCoreContext* core, VkGPUContext* gpu, VkFbo* fbo, VkSemaphore signalSemaphore, VkFence signalFence) {
    
    u32 imgIndex;
    auto top = core->vkScratch.top;

    img_acquire:
    VkResult imgAcquireResult = vkAcquireNextImageKHR(gpu->logicalDevice, fbo->swapChain, ~u64(0), signalSemaphore, signalFence, &imgIndex);
    if(imgAcquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        return ~u32(0);
    }
    core->vkScratch.top = top;

    return imgIndex;
}


void DestroyTexture(VkCoreContext* core, VkGPUContext* gpu, VkTextureInfo texture) {
    VK_CALL(core->vkScratch, vkDestroyImageView, gpu->logicalDevice, texture.view, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyImage, gpu->logicalDevice, texture.img, &core->vkAllocator);
}
void OnDrawFlatRetireCapture(VkRenderContext* ctx, void* resources) {

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
void RecordCopyBarrier(VkCoreContext* core, VkGPUContext* gpu, CmdState* cmdState, VkAccessFlags dstMask, VkAccessFlags dstStage, MemBlock dst, void* src) {

    VkBufferCopy copy{};
    copy.size = dst.size;
    copy.srcOffset = (byte*)src - gpu->uploadMemory.base;
    copy.dstOffset = dst.offset;
    VK_CALL(core->vkScratch, vkCmdCopyBuffer, cmdState->cmd, gpu->hostBuffer, gpu->deviceBuffer, 1, &copy);

    VkBufferMemoryBarrier descriptorBarrier{};
    descriptorBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    descriptorBarrier.buffer = gpu->deviceBuffer;
    descriptorBarrier.offset = copy.dstOffset;
    descriptorBarrier.size = copy.size;
    descriptorBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    descriptorBarrier.dstAccessMask = dstMask;
    descriptorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    descriptorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, cmdState->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage, 0, 0, nullptr, 1, &descriptorBarrier, 0, nullptr);

}
void RecordRender(VkCoreContext* core, VkGPUContext* gpu, CmdState cmd, u32 fboImg, MemBlock block, DrawflatCmdResources* resources, DrawInfo* draws, u32 drawCount, bool screenShot) {

    VkBufferCopy copy{};
    copy.size = resources->allocations[0].size;
    copy.srcOffset = (byte*)resources->allocations[0].ptr - gpu->uploadMemory.base;
    copy.dstOffset = 0;
    VK_CALL(core->vkScratch, vkCmdCopyBuffer, cmd.cmd, gpu->hostBuffer, gpu->deviceBuffer, 1, &copy);

    VkBufferMemoryBarrier descriptorBarrier{};
    descriptorBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    descriptorBarrier.buffer = gpu->deviceBuffer;
    descriptorBarrier.offset = copy.dstOffset;
    descriptorBarrier.size = copy.size;
    descriptorBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    descriptorBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    descriptorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    descriptorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, cmd.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &descriptorBarrier, 0, nullptr);

    VkBufferCopy instanceCpy{};
    instanceCpy.size = resources->allocations[1].size;
    instanceCpy.srcOffset = (byte*)resources->allocations[1].ptr - gpu->uploadMemory.base;
    instanceCpy.dstOffset = block.offset;
    VK_CALL(core->vkScratch, vkCmdCopyBuffer, cmd.cmd, gpu->hostBuffer, gpu->deviceBuffer, 1, &instanceCpy);

    VkBufferMemoryBarrier instanceBarrier{};
    instanceBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    instanceBarrier.buffer = gpu->deviceBuffer;
    instanceBarrier.offset = instanceCpy.dstOffset;
    instanceBarrier.size = instanceCpy.size;
    instanceBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    instanceBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    instanceBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    instanceBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VK_CALL(core->vkScratch, vkCmdPipelineBarrier, cmd.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &instanceBarrier, 0, nullptr);
    /*
    if(screenShot) {

        resources->engine = state;
        VkImageMemoryBarrier imgBarrier{};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.image = ctx->swapChainFrames[fboImg].colorImg;
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
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdPipelineBarrier, cmd.cmd, srcStage, dstStage, 0,  0, nullptr, 0, nullptr, 1, &imgBarrier);

        resources->allocationCount;
        resources->allocations[2].size = ctx->width * ctx->height * 4;
        resources->allocations[2].ptr = linear_alloc(&ctx->uploadMemory, ctx->width * ctx->height * 4);

        VkBufferImageCopy region{};
        region.imageExtent = {ctx->width, ctx->height, 1};
        region.imageOffset = {0,0,0};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.bufferImageHeight = ctx->height;
        region.bufferOffset = (byte*)resources->allocations[2].ptr - ctx->uploadMemory.base;
        region.bufferRowLength = ctx->width;
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdCopyImageToBuffer, cmd.cmd, ctx->swapChainFrames[fboImg].colorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ctx->hostBuffer, 1, &region);

        imgBarrier = {};
        imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imgBarrier.image = ctx->swapChainFrames[fboImg].colorImg;
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
        VK_CALL(ctx->vkCtx.vkScratch, vkCmdPipelineBarrier, cmd.cmd, srcStage, dstStage, 0,  0, nullptr, 0, nullptr, 1, &imgBarrier);
    }
    */
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
        debugMessengerCreateInfo.messageSeverity =  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        debugMessengerCreateInfo.messageType =  VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugMessengerCreateInfo.pfnUserCallback = DebugCallback;
        debugMessengerCreateInfo.pUserData = dst;
        createInfo.pNext = &debugMessengerCreateInfo;
    }

    auto requiredExt = (const char**)linear_allocator_top(&dst->vkScratch);
    createInfo.enabledExtensionCount = GetRequiredExtensions(requiredExt, INSTANCE_EXTENSIONS, SIZE_OF_ARRAY(INSTANCE_EXTENSIONS));
    createInfo.ppEnabledExtensionNames = requiredExt;
    dst->vkScratch.top += sizeof(const char**) * createInfo.enabledExtensionCount;

    VK_CALL(dst->vkScratch, vkCreateInstance, &createInfo, &dst->vkAllocator, &dst->vkInstance);
    dst->vkScratch.top -= sizeof(const char**) * createInfo.enabledExtensionCount;

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
    ret.families = GetQueueFamilies(core, ret.device, surface);
    ret.logicalDevice = CreateLogicalDevice(core, ret.families, ret.device);
    
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.graphicsFamily, 0, &ret.graphicsQueue);
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.presentFamily,  0, &ret.presentQueue);
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.computeFamily,  0, &ret.computeQueue);
    VK_CALL(core->vkScratch, vkGetDeviceQueue, ret.logicalDevice, ret.families.transferFamily, 0, &ret.transferQueue);

    mem = linear_allocate(alloc, config.uploadHeapSize);
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

    ret.descriptorUpdates = {};
    ret.cpuCmdBuffer = make_circular_allocator(linear_allocate(alloc, Kilobyte(4)), Kilobyte(4));

    return ret;
}
VkFbo MakeVkFbo(VkCoreContext* core, VkGPUContext* gpu, VkRenderPass renderPass, VkSurfaceKHR surface, xcb_connection_t* connection, xcb_context* xcb_ctx, Printer printer, LinearAllocator* alloc) {

    VkFbo ret{};

    ret.frameCount = 3;
    ret.surface = surface;
    auto res = CreateSwapChain(core, 3, gpu->device, gpu->logicalDevice, ret.surface, {xcb_ctx->width, xcb_ctx->height}, gpu->families, nullptr);
    ret.width = res.dims.x;
    ret.height = res.dims.y;
    ret.swapChain = res.swapChain;
    ret.fboColorFormat = res.format;
    ret.depthFormat = FindDepthFormat(core, gpu->device);
    ret.frames = (SwapChainFrame*)linear_allocate(alloc, ret.frameCount * sizeof(SwapChainFrame));
    
    u32 imageCount = 0;
    VK_CALL(core->vkScratch, vkGetSwapchainImagesKHR, gpu->logicalDevice, ret.swapChain, &imageCount, nullptr);
    VkImage images[imageCount];
    VK_CALL(core->vkScratch, vkGetSwapchainImagesKHR, gpu->logicalDevice, ret.swapChain, &imageCount, images);

    u32 families[2] = {gpu->families.graphicsFamily, gpu->families.presentFamily};
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
/*
VkProgram MakeVKProgram(VkCoreContext* core, VkGPUContext* gpu, VkPrgoramDescriptor description, Printer printer, LinearAllocator* alloc) {

    VkProgram ret{};
    for(u32 i = 0; i < description.subpassCount; i++) {
        description.subpasses[i];
    }

}
*/
VkProgram MakeVKProgram(VkCoreContext* core, VkGPUContext* gpu, VkPrgoramDescriptor description, Printer printer, LinearAllocator* alloc) {

    VkProgram ret{};

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
    AllocateDescriptorSets(core, gpu->logicalDevice, gpu->descriptorPool, 3, ret.UBOdescriptorLayout, 0, sets);
    AllocateDescriptorSets(core, gpu->logicalDevice, gpu->descriptorPool, 1, ret.textureDescriptorLayout, 512, &ret.textureDescriptors);

    VkWriteDescriptorSet descriptorWrites[3];
    VkDescriptorBufferInfo bufferInfo[3];
    ret.descriptorSetPool.resourceCount = 3;
    ret.descriptorSetPool.resources = (Descriptor*)linear_allocate(alloc, sizeof(Descriptor) * ret.descriptorSetPool.resourceCount);

    for(u32 i = 0; i < ret.descriptorSetPool.resourceCount; i++) {
        
        auto block = allocate_gpu_block(&gpu->gpuAllocator, sizeof(CommonParams), 256);
        ret.descriptorSetPool.resources[i].offset = block.offset;
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

    VkDescriptorSetLayout descriptorLayouts[2] = {ret.UBOdescriptorLayout, ret.textureDescriptorLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = descriptorLayouts;
    VK_CALL(core->vkScratch, vkCreatePipelineLayout, gpu->logicalDevice, &pipelineLayoutInfo, &core->vkAllocator, &ret.pipelineLayout);

    ret.renderPass = CreateRenderPass(core, gpu->logicalDevice, description.colorFormat, description.depthFormat);
    ret.graphicsPipeline = CreateGraphicsPipeline(core, gpu->logicalDevice, ret.pipelineLayout, ret.renderPass, description.width, description.height);

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
void DestroyVkProgram(VkCoreContext* core, VkGPUContext* gpu, VkProgram* program) {

    VK_CALL(core->vkScratch, vkDestroySampler, gpu->logicalDevice, program->textureSampler, &core->vkAllocator);

    VK_CALL(core->vkScratch, vkDestroyDescriptorSetLayout, gpu->logicalDevice, program->UBOdescriptorLayout, &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyDescriptorSetLayout, gpu->logicalDevice, program->textureDescriptorLayout, &core->vkAllocator);

    VK_CALL(core->vkScratch, vkDestroyPipeline,         gpu->logicalDevice, program->graphicsPipeline,  &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyRenderPass,       gpu->logicalDevice, program->renderPass,        &core->vkAllocator);
    VK_CALL(core->vkScratch, vkDestroyPipelineLayout,   gpu->logicalDevice, program->pipelineLayout,    &core->vkAllocator);
}
#pragma once
#include <type_traits>
#include <common.h>

template<typename Callee, typename ... Args> void VK_CHECK_IF_NOT_VOID(int line, const char* file, Callee fn, Args ... args) {

    if constexpr (!std::is_same<void, decltype(fn(args...)) >::value) {
        auto res = fn(args...);
        if (res != VK_SUCCESS) {
            global_print("sissc", "VK_CALL failed at ", line, " in file ", file , '\n');
            global_io_flush();
            _TRAP;
        }
    }
    else {
        fn(args...);
    }
}

constexpr bool ENABLE_VALIDATION_LAYER = true;
#if VULKAN_DEBUG == 1
    #define GLFW_CALL(x) GLFWclearError(); x; ASSERT(GLFWlogCall(#x , __FILE__ , __LINE__));
    #define VK_CALL(scratchMemory, ...) {                        \
        auto save = scratchMemory.top;                           \
        VK_CHECK_IF_NOT_VOID(__LINE__, __FILE__, __VA_ARGS__);   \
        scratchMemory.top = save;                                \
    }

#else
    #define GLFW_CALL(x) x
    #define ARGS(c, ...) __VA_ARGS__
    #define CALLEE(c, ...) c
    #define VK_CALL(scratchMemory, ...) {           \
        auto save = scratchMemory.top;              \
        CALLEE(__VA_ARGS__)(ARGS(__VA_ARGS__));     \
        scratchMemory.top = save;                   \
    }
#endif

// void GLFWclearError();
// bool GLFWlogCall(const char* function, const char* file, int line);
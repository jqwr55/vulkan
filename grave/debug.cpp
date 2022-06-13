/*
#include <debug.h>
#include <common.h>
#include <vulkan/vulkan.h>

void GLFWclearError() {
    const char* description;
    while(glfwGetError(&description) != GLFW_NO_ERROR );
}

bool GLFWlogCall(const char* function, const char* file, int line) {
    const char* description;
    while(auto error = glfwGetError(&description) ) {

        char* errorStr;
        switch (error) {
            case GLFW_NO_ERROR:
                errorStr = "GLFW_NO_ERROR";
                break;

            case GLFW_NOT_INITIALIZED:
                errorStr = "GLFW_NOT_INITIALIZED";
                break;

            case GLFW_NO_CURRENT_CONTEXT:
                errorStr = "GLFW_NO_CURRENT_CONTEXT";
                break;

            case GLFW_INVALID_ENUM:
                errorStr = "GLFW_INVALID_ENUM";
                break;

            case GLFW_INVALID_VALUE:
                errorStr = "GLFW_INVALID_VALUE";
                break;

            case GLFW_OUT_OF_MEMORY:
                errorStr = "GLFW_OUT_OF_MEMORY";
                break;

            case GLFW_API_UNAVAILABLE:
                errorStr = "GLFW_API_UNAVAILABLE";
                break;

            case GLFW_VERSION_UNAVAILABLE:
                errorStr = "GLFW_VERSION_UNAVAILABLE";
                break;

            case GLFW_PLATFORM_ERROR:
                errorStr = "GLFW_PLATFORM_ERROR";
                break;

            case GLFW_FORMAT_UNAVAILABLE :
                errorStr = "GLFW_FORMAT_UNAVAILABLE";
                break;

            case GLFW_NO_WINDOW_CONTEXT  :
                errorStr = "GLFW_NO_WINDOW_CONTEXT";
                break;


            default:
                errorStr = "unkown error code";
                break;
        }

        global_print("sscscicscsc", "[GLFW] runtime error: ", errorStr, ' ', function, ' ', (i64)line, ' ', file, ' ', description, '\n');
        return false;
    }
    return true;
}

*/
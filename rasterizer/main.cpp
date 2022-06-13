#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <interface.h>
#include <debug.h>
#include <common.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_true_type.h>
#include <open_type_loader.h>
#include <math3d.h>

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
void DrawLine(Pixel* dst, u32 width, u32 height, Pixel col, vec<f32,2> p0, vec<f32,2> p1) {

    vec<f32, 2> dist = p1 - p0;
    auto delta = normalize(dist);
    f32 distSquared = dist.x * dist.x + dist.y * dist.y;

    while(distSquared > 1) {

        u32 index = (u32)p0.y * width + (u32)p0.x;
        index = Clamp(index, (u32)0, width * height);
        dst[index] = col;
        p0 = p0 + delta;

        dist = {
            (p1.x - p0.x),
            (p1.y - p0.y),
        };
        distSquared = dist.x * dist.x + dist.y * dist.y;
    }
}

constexpr u32 WIDTH = 640;
constexpr u32 HEIGHT = 480;
GLFWwindow* Init() {

    if (!glfwInit()) {
        global_print("s", "Failed to initialize GLFW\n");
        global_io_flush();
        return nullptr;
    }
    sleep(2);

    GLFW_CALL(glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE));
    sleep(1);
    GLFW_CALL(GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "rasterizer", NULL, NULL));
    if (!window) {
        glfwTerminate();
        global_print("s", "Failed to create glfw window\n");
        global_io_flush();
        return nullptr;
    }
    sleep(6);

    GLFW_CALL(glfwMakeContextCurrent(window));
    sleep(1);
    GLFW_CALL(glfwSwapInterval(2));
    sleep(1);

    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        global_print("s", "Failed to initialize glew\n");
        global_io_flush();
        return nullptr;
    }
    sleep(1);

    return window;
}
void DrawTriangles(Pixel* dst, u32 width, u32 height, u32 triangleCount, i32 triangles[], v2* vertices) {

    for(u32 i = 0; i < triangleCount; i++) {

        auto p0 = vertices[triangles[i * 3 + 0] - 1];
        auto p1 = vertices[triangles[i * 3 + 1] - 1];
        auto p2 = vertices[triangles[i * 3 + 2] - 1];

        vec<f32, 2> q0 = {p0.x, p0.y};
        vec<f32, 2> q1 = {p1.x, p1.y};
        vec<f32, 2> q2 = {p2.x, p2.y};
        DrawLine(dst, width, height, {255,255,255,255}, q0, q1);
        DrawLine(dst, width, height, {255,255,255,255}, q1, q2);
        DrawLine(dst, width, height, {255,255,255,255}, q2, q0);
    }
}
void DrawEdges(Pixel* framebuffer, u32 width, u32 height, u32 edgeCount, Edge* edges) {

    for(u32 i = 0; i < edgeCount-1; i++) {

        vec<f32,2> p0 {
            edges[i].p0.x * 300.f,
            edges[i].p0.y * 300.f
        };
        vec<f32,2> p1 {
            edges[i].p1.x * 300.f,
            edges[i].p1.y * 300.f
        };
        p0 = p0 + vec<f32,2>{100.0f, 100.0f};
        p1 = p1 + vec<f32,2>{100.0f, 100.0f};
        DrawLine(framebuffer, width, height, {255,255,255,255}, p0, p1);
    }
}
void DrawLines(Pixel* framebuffer, u32 width, u32 height, u32 vCount, v2* lines) {

    for(u32 i = 0; i < vCount-1; i++) {

        vec<f32,2> p0 {
            lines[i].x,
            lines[i].y
        };
        vec<f32,2> p1 {
            lines[i+1].x,
            lines[i+1].y
        };
        DrawLine(framebuffer, width, height, {255,255,255,255}, p0, p1);
    }

    vec<f32,2> p0 {
        lines[vCount-1].x,
        lines[vCount-1].y
    };
    vec<f32,2> p1 {
        lines[0].x,
        lines[0].y
    };
    DrawLine(framebuffer, width, height, {255,255,255,255}, p0, p1);
}
void ApplyTransform(v2* src, u32 count, v2 scale, v2 offset) {

    for(u32 i = 0; i < count; i++) {
        src[i].x = (src[i].x * scale.x) + offset.x;
        src[i].y = (src[i].y * scale.y) + offset.y;
    }
}
void TransformtoNDC(v2* src, u32 count) {

    f32 maxX = 0;
    f32 maxY = 0;
    for(u32 i = 0; i < count; i++) {
        maxX = Max(maxX, src[i].x);
        maxY = Max(maxY, src[i].y);
    }
    f32 scaleX = 1.0f / maxX;
    f32 scaleY = 1.0f / maxY;
    for(u32 i = 0; i < count; i++) {
        src[i].x *= scaleX;
        src[i].y *= scaleY;
    }
}
void DrawPoint(Pixel* dst, u32 width, u32 height, Pixel col, v2 p) {
    u32 x = Clamp((u32)p.x, (u32)0, width);
    u32 y = Clamp((u32)p.y, (u32)0, height);

    u32 i = (u32)y * width + (u32)x;
    dst[i] = col;
}
void DrawPointThick(Pixel* dst, u32 width, u32 height, Pixel col, v2 p) {

    DrawPoint(dst, width, height, col, p + v2{0,0});
    DrawPoint(dst, width, height, col, p + v2{1,0});
    DrawPoint(dst, width, height, col, p + v2{0,1});

    DrawPoint(dst, width, height, col, p + v2{1,1});
    DrawPoint(dst, width, height, col, p - v2{1,0});
    DrawPoint(dst, width, height, col, p - v2{0,1});

    DrawPoint(dst, width, height, col, p - v2{1,1});
    DrawPoint(dst, width, height, col, p + v2{1,-1});
    DrawPoint(dst, width, height, col, p + v2{-1, 1});
}
struct FrameBuffer {
    Pixel* dst;
    u32 w;
    u32 h;
};

u32 GetInterections(FrameBuffer fbo, u32 count, v2* poly, v2 rayo, v2 rayd, LinearAllocator* result) {

    u32 ret = 0;
    for(u32 i = 0; i < count-1; i++) {

        v2 r = RayLineSegmentIntersection(rayo, rayd, poly[i], poly[i+1]);
        if(r != v2{0.0,0.0} && r != rayo) {

            v2* intersect = (v2*)linear_allocate(result, sizeof(v2));
            *intersect = r;
            ret++;
        }

    }

    v2 r = RayLineSegmentIntersection(rayo, rayd, poly[0], poly[count-1]);
    if(r != v2{0.0,0.0} && r != rayo) {

        v2* intersect = (v2*)linear_allocate(result, sizeof(v2));
        *intersect = r;
        ret++;
    }

    return ret;
}
v2 GetClosestInterection(FrameBuffer fbo, u32 count, v2* poly, v2 rayo, v2 rayd) {

    v2 closestHit = {0.0,0.0};
    f32 distSquared = 1/0.0f;

    u32 seg = ~u32(0);

    for(u32 i = 0; i < count-1; i++) {

        if(poly[i] == rayo || poly[i+1] == rayo) continue;

        v2 r = RayLineSegmentIntersection(rayo, rayd, poly[i], poly[i+1]);
        if(r != v2{0.0,0.0} && r != rayo) {
            v2 tmp = r - rayo;
            auto dist = dot(tmp, tmp);
            if(dist < distSquared) {
                closestHit = r;
                distSquared = dist;

                seg = i;
            }
        }
    }


    if(poly[0] != rayo && poly[count - 1] != rayo) {

        v2 r = RayLineSegmentIntersection(rayo, rayd, poly[0], poly[count - 1]);
        if(r != v2{0.0,0.0} && r != rayo) {
            v2 tmp = r - rayo;
            auto dist = dot(tmp, tmp);
            if(dist < distSquared) {
                closestHit = r;
                distSquared = dist;
                seg = count - 1;
            }
        }
    }
    if(seg != ~u32(0)) {
        if(seg == count - 1) {
            DrawLine(fbo.dst, fbo.w, fbo.h, {0,0,255,255}, poly[0], poly[count - 1]);
        }
        else {
            DrawLine(fbo.dst, fbo.w, fbo.h, {0,0,255,255}, poly[seg], poly[seg+1]);
        }
    }

    return closestHit;
}
i32 edgeFunction(v2 a, v2 b, v2 c) {
    return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
}
void DrawTriangle(FrameBuffer fbo, Pixel col, v2* vertices, u32* indices) {

    v2 v[3];
    v[0] = vertices[indices[0] - 1];
    v[1] = vertices[indices[1] - 1];
    v[2] = vertices[indices[2] - 1];

    i32 minx = Min<i32>(v[0].x, v[1].x);
    minx     = Min<i32>(minx, v[2].x);

    i32 miny = Min<i32>(v[0].y, v[1].y);
    miny     = Min<i32>(miny, v[2].y);

    i32 max_x = Max<i32>(v[0].x, v[1].x);
    max_x     = Max<i32>(max_x, v[2].x);

    i32 max_y = Max<i32>(v[0].y, v[1].y);
    max_y     = Max<i32>(max_y, v[2].y);

    minx = Max<i32>(minx,0);
    minx = Min<i32>(minx, fbo.w);

    miny = Max<i32>(miny,0);
    miny = Min<i32>(miny,fbo.h);

    max_x = Max<i32>(max_x,0);
    max_x = Min<i32>(max_x,fbo.w);

    max_y = Max<i32>(max_y,0);
    max_y = Min<i32>(max_y, fbo.h);

    for(;miny < max_y;miny++) {
        u32 index = fbo.w * miny;
        float u = 0;
        for(i32 x = minx;x < max_x;x++) {

            i32 t0 = edgeFunction(v[1] ,v[2] , {x,miny} );
            if( t0 < 0 ) {
                continue;
            }
            i32 t1 = edgeFunction(v[2] ,v[0] , {x,miny} );
            if( t1 < 0 ) {
                continue;
            }
            i32 t2 = edgeFunction(v[0] ,v[1] , {x,miny} );
            if( t2 < 0 ) {
                continue;
            }

            fbo.dst[fbo.w * miny + x] = col;
        }
    }
}

void RasterizeTriangle(FrameBuffer fbo , Pixel col, v2* vertices, u32* indices) {

    v2 v[3];
    v[0] = vertices[indices[0] - 1];
    v[1] = vertices[indices[1] - 1];
    v[2] = vertices[indices[2] - 1];

    i32 minx = Min<i32>(v[0].x, v[1].x);
    minx     = Min<i32>(minx, v[2].x);

    i32 miny = Min<i32>(v[0].y, v[1].y);
    miny     = Min<i32>(miny, v[2].y);

    i32 max_x = Max<i32>(v[0].x, v[1].x);
    max_x     = Max<i32>(max_x, v[2].x);

    i32 max_y = Max<i32>(v[0].y, v[1].y);
    max_y     = Max<i32>(max_y, v[2].y);

    minx = Max<i32>(minx,0);
    minx = Min<i32>(minx, fbo.w);

    miny = Max<i32>(miny,0);
    miny = Min<i32>(miny,fbo.h);

    max_x = Max<i32>(max_x,0);
    max_x = Min<i32>(max_x,fbo.w);

    max_y = Max<i32>(max_y,0);
    max_y = Min<i32>(max_y, fbo.h);


    f32 dx0 = (v[1].y - v[0].y);
    f32 dx1 = (v[2].y - v[1].y);
    f32 dx2 = (v[0].y - v[2].y);

    f32 dy0 = (v[1].x - v[0].x);
    f32 dy1 = (v[2].x - v[1].x);
    f32 dy2 = (v[0].x - v[2].x);

    f32 w0 = dy0 * ((f32)miny - v[0].y) - dx0 * ((f32)minx - v[0].x);
    f32 w1 = dy1 * ((f32)miny - v[1].y) - dx1 * ((f32)minx - v[1].x);
    f32 w2 = dy2 * ((f32)miny - v[2].y) - dx2 * ((f32)minx - v[2].x);

    f32 r0 = dx0 * ((f32)max_x - (f32)minx) + dy0;
    f32 r1 = dx1 * ((f32)max_x - (f32)minx) + dy1;
    f32 r2 = dx2 * ((f32)max_x - (f32)minx) + dy2;

    for(;miny < max_y ; miny++) {
        for(uint x = minx ; x < max_x;x++) {

            if( (w0 >= 0) & (w1 >= 0) & (w2 >= 0) ) {
            // if( (w0 | w1 | w2) > 0 ) {

                fbo.dst [x + fbo.w * miny] = col;
            }
            w0 -= dx0;
            w1 -= dx1;
            w2 -= dx2;
        }
        w0 += r0;
        w1 += r1;
        w2 += r2;
    }
}

i32 main() {

    auto mem = init_global_state(0, Megabyte(16), 512);
    auto alloc = make_linear_allocator(mem, Megabyte(16));

    Pixel* framebuffer = (Pixel*)linear_allocate(&alloc, WIDTH * HEIGHT * 4);
    memset(framebuffer, 0, WIDTH * HEIGHT * 4);
    FrameBuffer fbo{
        .dst = framebuffer,
        .w = WIDTH,
        .h = HEIGHT,
    };

    byte* fontBase = (byte*)linear_allocator_top(&alloc);
    auto fontSize = ReadFile("/usr/share/fonts/opentype/malayalam/Manjari-Regular.otf", &alloc);

    f32 height = 300.0f;
    auto info = ParseOTFMemory(fontBase, fontSize, &alloc);
    f32 scale = GetScaleForPixelHeight(&info, height / 10);
    auto outline = ExtractGlyphOutlineOTF(&info, 'B', &alloc);

    u32 contours[outline.contourCount];
    v2* poly = (v2*)linear_allocator_top(&alloc);
    u32 pCount = TesselateOutline(outline, contours, 0.35f / (scale*10), &alloc);

    {
        i32 triangleList[200][3]{};
        auto triangleCount = TriangulateTesselatedContours(triangleList, outline.contourCount, contours, pCount, poly, &alloc);

        // v2 poly2[] = {
        //     {100.0, 100.0}, {150.0f, 150.0f}, {150.0f, 200.0f}, {90.0f, 180.0f}, {40.0f, 120.0f}
        // };
        // contours[0] = 5;
        // auto triangleCount2 = TriangulateTesselatedContours(triangleList, 1, contours, 5, poly2, &alloc);

        TransformtoNDC(poly, pCount);
        ApplyTransform(poly, pCount, {height, height}, {100.0f, 100.0f} );

        for(u32 i = 0; i < triangleCount; i++) {

            RasterizeTriangle(fbo, {255,0,0,0}, poly, (u32*)(triangleList + i));
        }
        //DrawTriangles(framebuffer, WIDTH, HEIGHT, triangleCount, (i32*)triangleList, poly);
    }

    v2 rayo{100.0f, 100.0f};
    v2 raydir = normalize(v2{100.0f,100.0f});
    v2 a{300.0f, 80.0f};
    v2 b{100.0f, 280.0f};
    auto window = Init();
    ASSERT(window);
    sleep(2);

    i32 co = 0;
    i32 po = 0;
    while(!glfwWindowShouldClose(window)) {

        usleep(200000);

        GLFW_CALL(glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, framebuffer));
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();

    return 0;
}

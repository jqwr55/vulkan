#include "common.h"

struct vlx_u16_v2 {
    u16 x;
    u16 y;
};
struct vlx_rect {
    vlx_u16_v2 left_bottom;
    vlx_u16_v2 right_top;
};
struct vlx_color {
    u8 r,g,b,a;
};
struct vlx_image {

    u64 handle;
    vlx_rect view;
};
struct vlx_style_info {

};
struct vlx_font_info {

};
struct vlx_vertex_attrib_descriptor {
    u16 attrib;
    u16 format;
    u16 offset;
};
struct vlx_context {
    
    vlx_rect view;
    vlx_u16_v2 cursor;
    vlx_u16_v2 prev_cursor;
    u32 cursor_button_state;
    vlx_style_info style;
    vlx_font_info font;

    u32 attribCount;
    u32 vertex_size;
    vlx_vertex_attrib_descriptor* attribs;

    void* hover;
    void* active;
    void* focused;

    LocalMallocState heap;
};

struct vlx_button {
    vlx_rect bounds;
    const char* text;
};
enum vlx_flags_enum : u32 {

    VLX_FLAG_NONE          = 0,
    VLX_FLAG_VISIBLE_BIT   = 1 << 0,
    VLX_FLAG_BUFFER_FULL   = 1 << 1,
    VLX_FLAG_HOVER_BIT     = 1 << 2,
    VLX_FLAG_ACTIVE_BIT    = 1 << 3,
    VLX_FLAG_INTERACT_BIT  = 1 << 4,
};
typedef u32 vlx_flags;
struct vlx_window {
   
    vlx_rect bounds;
    vlx_flags state;
    vlx_flags properties;

    u32 widget_mem_top;
    u32 widget_mem_size;
    byte widget_mem[];
};

struct vlx_geometry_alloc {
    LinearAllocator shapes;
    LinearAllocator vertex;
    LinearAllocator index;
};

bool vlx_point_inside_rect(vlx_u16_v2 p, vlx_rect rect) {

    return  rect.left_bottom.x >= p.x   &&
            rect.left_bottom.y >= p.y   &&
            rect.right_top.x <= p.x     &&
            rect.right_top.y <= p.y;
}
bool vlx_rect_overlap(vlx_rect r0, vlx_rect r1) {

    bool b = r0.left_bottom.x > r1.right_top.x ||
             r1.left_bottom.x > r0.right_top.x ||
             r0.left_bottom.y > r1.right_top.y ||
             r1.left_bottom.x > r0.right_top.y;
    return !b;
}

bool vlx_push_quad(vlx_geometry_alloc* dst, vlx_rect rect, vlx_vertex_attrib_descriptor* layout) {

}
vlx_flags vlx_update_interaction(vlx_context* ctx, vlx_rect bounds, void* obj) {

    vlx_flags f = 0;
    bool inside = vlx_point_inside_rect(ctx->cursor, bounds);

    if(inside) {
        f |= VLX_FLAG_HOVER_BIT;
        ctx->hover = obj;
    }
    if(ctx->active == obj && !ctx->cursor_button_state) {
    
        if (inside) {
            f |= VLX_FLAG_INTERACT_BIT;
            ctx->hover = obj;
        }
        else {
            
            ctx->hover  = nullptr;
            ctx->active = nullptr;
        }
    }
    else if (ctx->hover == obj) {

        if (ctx->cursor_button_state) {

            f |= VLX_FLAG_ACTIVE_BIT;
            ctx->focused = obj;
            ctx->active  = obj;
        }
    }

    return f;
}

vlx_window* vlx_begin_window_vertex(vlx_context* ctx, vlx_window* win, vlx_rect bounds, vlx_color, vlx_image image, vlx_flags properties, vlx_geometry_alloc* geometry) {

    if(!win) {

        win = (vlx_window*)local_malloc(&ctx->heap, sizeof(vlx_window) + 1000);
        *win = {};
        win->bounds = bounds;
        win->properties = properties;
        win->widget_mem_size = 1000;
    }

    vlx_flags f = vlx_update_interaction(ctx, win->bounds, win);
    if(f & VLX_FLAG_ACTIVE_BIT) {

        u32 width = win->bounds.left_bottom.x - win->bounds.right_top.x;
        u32 height = win->bounds.left_bottom.y - win->bounds.right_top.y;

        vlx_rect right_edge {
            .left_bottom = {win->bounds.right_top.x - 5, win->bounds.right_top.y - height},
            .right_top = win->bounds.right_top,
        };

        u16 delta_x = ctx->cursor.x - ctx->prev_cursor.x;
        u16 delta_y = ctx->cursor.y - ctx->prev_cursor.y;

        win->bounds.left_bottom.x += delta_x;
        win->bounds.left_bottom.y += delta_y;

        win->bounds.right_top.x += delta_x;
        win->bounds.right_top.y += delta_y;
    }

    vlx_push_quad(geometry, win->bounds, ctx->attribs);
    return win;
}

vlx_flags vlx_do_button_vertex(vlx_context* ctx, vlx_window* win, vlx_rect bounds, const char* text, vlx_geometry_alloc* geometry) {

    vlx_button* button = (vlx_button*)(win->widget_mem + win->widget_mem_top);
    win->widget_mem_top += sizeof(vlx_button);

    button->bounds = bounds;
    button->text = text;

    vlx_flags result = vlx_update_interaction(ctx, button->bounds, button);
    vlx_push_quad(geometry, button->bounds, ctx->attribs);

    return result;
}

vlx_context make_vlx_context(void* mem, u32 size) {

    vlx_context ret{};
    ret.heap = make_local_malloc((byte*)mem, size);

    return ret;
}
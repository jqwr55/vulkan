#pragma once
#include <common.h>
#include <xcb/xcb.h>

struct xcb_context {

    xcb_screen_t*           screen;
    xcb_window_t            window;
    xcb_intern_atom_reply_t window_delete;

    u32 width;
    u32 height;
    u32 cursorsX;
    u32 cursorsY;
    u32 keys;
    bool button0; // left
    bool button1; // right
    bool button2; // middle
    bool open;
};

xcb_context make_xcb_context(xcb_connection_t* connection, xcb_window_t id, u32 w, u32 h, const char* title);
void consume_xcb_events(xcb_connection_t* connection, u32 ctxCount, xcb_context* ctx);

enum KEY_MASK : u32 {
    KEY_BIT_W           = 0,
    KEY_BIT_A           = 1,
    KEY_BIT_S           = 2,
    KEY_BIT_D           = 3,
    KEY_BIT_SPACE       = 4,
    KEY_BIT_LEFT_SHIFT  = 5,
};
constexpr auto KEY_ESCAPE   = 0x9;
constexpr auto KEY_F1       = 0x43;
constexpr auto KEY_F2       = 0x44;
constexpr auto KEY_F3       = 0x45;
constexpr auto KEY_F4       = 0x46;
constexpr auto KEY_W        = 0x19;
constexpr auto KEY_A        = 0x26;
constexpr auto KEY_S        = 0x27;
constexpr auto KEY_D        = 0x28;
constexpr auto KEY_P        = 0x21;
constexpr auto KEY_SPACE    = 0x41;
constexpr auto KEY_KPADD    = 0x56;
constexpr auto KEY_KPSUB    = 0x52;
constexpr auto KEY_B        = 0x38;
constexpr auto KEY_F        = 0x29;
constexpr auto KEY_L        = 0x2E;
constexpr auto KEY_N        = 0x39;
constexpr auto KEY_O        = 0x20;
constexpr auto KEY_T        = 0x1C;
constexpr auto KEY_SHIFT    = 50;
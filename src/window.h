#pragma once
#include "common.h"
#include <xcb/xcb.h>
#include <xkbcommon/xkbcommon.h>
#include <xkbcommon/xkbcommon-compose.h>
#include <xkbcommon/xkbcommon-x11.h>

struct xcb_context {

    xcb_screen_t*   screen;
    xcb_window_t    window;
    xcb_atom_t      window_delete;
    xcb_atom_t      wm_state;
    xcb_atom_t      wm_state_hidden;

    u32 width;
    u32 height;
    u32 cursors_x;
    u32 cursors_y;
    u32 keys;

    DynamicBufferDebugMalloc<xkb_keysym_t> keySymbolBuffer;
    bool button0; // left
    bool button1; // right
    bool button2; // middle
    bool open;
};

struct xkb_keyboard {
    xkb_context* ctx;
    xkb_keymap* keymap;
    xkb_state* state;
    i32 device_id;
    char key_str[64];
};

xcb_context make_xcb_context(xcb_connection_t* connection, xcb_window_t id, u32 w, u32 h, const char* title);
void consume_xcb_events(xcb_connection_t* connection, xkb_keyboard* keyboard, u32 ctxCount, xcb_context* ctx);
xkb_keyboard make_xcb_keys(xcb_connection_t* connection);

enum KEY_MASK : u32 {
    KEY_BIT_W           = 0,
    KEY_BIT_A           = 1,
    KEY_BIT_S           = 2,
    KEY_BIT_D           = 3,
    KEY_BIT_SPACE       = 4,
    KEY_BIT_LEFT_SHIFT  = 5,
    KEY_BIT_LEFT_CTRL   = 6,
};
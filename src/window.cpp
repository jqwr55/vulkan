#include <window.h>

xkb_keyboard make_xcb_keys(xcb_connection_t* connection) {

    xkb_keyboard ret{};
    u8 first_xkb_event;
    ASSERT(xkb_x11_setup_xkb_extension(
        connection,
        XKB_X11_MIN_MAJOR_XKB_VERSION,
        XKB_X11_MIN_MINOR_XKB_VERSION,
        XKB_X11_SETUP_XKB_EXTENSION_NO_FLAGS,
        NULL, NULL, &first_xkb_event, NULL
    ));

    ret.ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    ASSERT(ret.ctx);
    ret.device_id = xkb_x11_get_core_keyboard_device_id(connection);
    ASSERT(ret.device_id != -1);

    ret.keymap = xkb_x11_keymap_new_from_device(ret.ctx, connection, ret.device_id, XKB_KEYMAP_COMPILE_NO_FLAGS);
    
    ret.state = xkb_x11_state_new_from_device(ret.keymap, connection, ret.device_id);
    ASSERT(ret.state);

    return ret;
}
xcb_context make_xcb_context(xcb_connection_t* connection, xcb_window_t id, u32 w, u32 h, const char* title) {

    xcb_context ret{};
    ret.width = w;
    ret.height = h;
    ret.window = id;
    ret.open = true;

    auto setup = xcb_get_setup(connection);
    xcb_screen_t* s = xcb_setup_roots_iterator(setup).data;
    u32 valueMask = XCB_EVENT_MASK_EXPOSURE         | XCB_EVENT_MASK_BUTTON_PRESS   |
                    XCB_EVENT_MASK_BUTTON_RELEASE   | XCB_EVENT_MASK_POINTER_MOTION |
                    XCB_EVENT_MASK_ENTER_WINDOW     | XCB_EVENT_MASK_LEAVE_WINDOW   |
                    XCB_EVENT_MASK_KEY_PRESS        | XCB_EVENT_MASK_KEY_RELEASE    |
                    XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_PROPERTY_CHANGE;

    ret.keySymbolBuffer.Init();

    xcb_keycode_iterator_t it = {
        .data = 0,
        .rem = setup->max_keycode,
        .index = 0,
    };
    xcb_keycode_next(&it);
    xcb_keycode_next(&it);

    ret.screen = s;
    xcb_create_window(
        connection,
        0, ret.window, s->root,
        0, 0, 640, 480, 0,
        XCB_WINDOW_CLASS_INPUT_OUTPUT,
        s->root_visual, XCB_CW_EVENT_MASK,
        &valueMask
    );

    xcb_intern_atom_cookie_t cookie = xcb_intern_atom(connection, 1, 12, "WM_PROTOCOLS");
    xcb_intern_atom_cookie_t cookie2 = xcb_intern_atom(connection, 0, 16, "WM_DELETE_WINDOW");

    xcb_intern_atom_cookie_t cookie3 = xcb_intern_atom(connection, 0, 14, "_NET_WM_STATE");
    xcb_intern_atom_cookie_t cookie4 = xcb_intern_atom(connection, 0, 21, "_NET_WM_STATE_HIDDEN");

    xcb_intern_atom_reply_t* reply = xcb_intern_atom_reply(connection, cookie, 0);
    xcb_intern_atom_reply_t* reply2 = xcb_intern_atom_reply(connection, cookie2, 0);
    xcb_intern_atom_reply_t* reply3 = xcb_intern_atom_reply(connection, cookie3, 0);
    xcb_intern_atom_reply_t* reply4 = xcb_intern_atom_reply(connection, cookie4, 0);

    ret.wm_state = reply3->atom;
    ret.wm_state_hidden = reply4->atom;

    xcb_change_property(connection, XCB_PROP_MODE_REPLACE, ret.window, reply->atom, 4, 32, 1, &reply2->atom);

    auto len = Max(0, (i32)str_len(title) - 1);
    xcb_change_property(connection, XCB_PROP_MODE_REPLACE, ret.window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, len, title);
    ret.window_delete = reply2->atom;

    xcb_map_window(connection, ret.window);
    xcb_flush(connection);
    
    free(reply);
    free(reply2);
    free(reply3);
    free(reply4);

    return ret;
}
void handle_xcb_error(u32 ctxCount, xcb_context* ctx, xcb_generic_error_t* error) {

    global_print("suc", "xcb error event ", (u32)error->error_code, '\n');
    auto beginID = ctx->window;
    switch(error->error_code) {
    case XCB_REQUEST:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_VALUE:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_WINDOW:
        {
            auto e = (xcb_window_error_t*)error;
            e->error_code;

            break;
        }
    case XCB_ATOM:
        {
            auto e = (xcb_atom_error_t*)error;
            e->bad_value;
            break;
        }
    case XCB_CURSOR:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_MATCH:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_ACCESS:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_ALLOC:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_ID_CHOICE:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    case XCB_IMPLEMENTATION:
        {
            auto e = (xcb_request_error_t*)error;
            break;
        }
    }
}

xcb_atom_t helper(xcb_connection_t* connection, xcb_context* ctx, xcb_atom_t atom) {

    xcb_get_property_cookie_t cookie = xcb_get_property(connection, false, ctx->window, atom, XCB_ATOM_ATOM, 0, 32);
    xcb_generic_error_t* err{nullptr};

    auto reply = xcb_get_property_reply(connection, cookie, &err);
    xcb_atom_t* value = (xcb_atom_t*)xcb_get_property_value(reply);
    return *value;
}
void handle_xcb_event(xcb_connection_t* connection, xkb_keyboard* keyboard, u32 ctxCount, xcb_context* ctx, xcb_generic_event_t* event) {

    auto beginID = ctx->window;
    switch(event->response_type & ~0x80) {
    case 0:
        {
            auto error = (xcb_generic_error_t*)event;
            handle_xcb_error(ctxCount, ctx, error);
            break;
        }
	case XCB_CLIENT_MESSAGE:
        {
            auto msg = (xcb_client_message_event_t*)event;
            auto index = msg->window - beginID;
            if( msg->data.data32[0] == ctx[index].window_delete) {
                global_print("suc", "Kill client ", msg->window, '\n');
                xcb_destroy_window(connection, msg->window);
                ctx[index].open = false;
            }
            break;
        }
	case XCB_MOTION_NOTIFY:
        {
            auto motion = (xcb_motion_notify_event_t*)event;
            ctx[motion->event - beginID].cursors_x = motion->event_x;
            ctx[motion->event - beginID].cursors_y = motion->event_y;
            // global_print("susus", "cursor(", motion->event_x, ", ", motion->event_y, ")\n");
            break;
        }
	case XCB_BUTTON_PRESS:
        {
            auto press = (xcb_button_press_event_t*)event;


            ctx[press->event - beginID].button0 |= press->detail == XCB_BUTTON_INDEX_1;
            ctx[press->event - beginID].button1 |= press->detail == XCB_BUTTON_INDEX_2;
            ctx[press->event - beginID].button2 |= press->detail == XCB_BUTTON_INDEX_3;
            break;
        }
	case XCB_BUTTON_RELEASE:
        {
            auto press = (xcb_button_press_event_t*)event;
            ctx[press->event - beginID].button0 &= !(press->detail == XCB_BUTTON_INDEX_1);
            ctx[press->event - beginID].button1 &= !(press->detail == XCB_BUTTON_INDEX_2);
            ctx[press->event - beginID].button2 &= !(press->detail == XCB_BUTTON_INDEX_3);
            break;
        }
	case XCB_KEY_PRESS:
        {
            auto keyEvent = (xcb_key_release_event_t*)event;
            auto win = ctx + (keyEvent->event - beginID);
            
            xkb_keycode_t keycode = keyEvent->detail;
            xkb_state_update_key(keyboard->state, keycode, XKB_KEY_DOWN);
            win->keys |= (keycode == 25) << KEY_BIT_W;
            win->keys |= (keycode == 38) << KEY_BIT_A;
            win->keys |= (keycode == 39) << KEY_BIT_S;
            win->keys |= (keycode == 40) << KEY_BIT_D;
            win->keys |= (keycode == 50) << KEY_BIT_LEFT_SHIFT;
            win->keys |= (keycode == 65) << KEY_BIT_SPACE;
            win->keys |= (keycode == 37) << KEY_BIT_LEFT_CTRL;

            auto keysym = xkb_state_key_get_one_sym(keyboard->state, keycode);
            //ctx[keyEvent->event - beginID].keySymbolBuffer.PushBack(keysym);

	        break;
        }
	case XCB_KEY_RELEASE:
        {
            auto keyEvent = (xcb_key_release_event_t*)event;
            auto win = ctx + (keyEvent->event - beginID);
            xkb_keycode_t keycode = keyEvent->detail;

            win->keys &= ~((keycode == 25) << KEY_BIT_W);
            win->keys &= ~((keycode == 38) << KEY_BIT_A);
            win->keys &= ~((keycode == 39) << KEY_BIT_S);
            win->keys &= ~((keycode == 40) << KEY_BIT_D);
            win->keys &= ~((keycode == 50) << KEY_BIT_LEFT_SHIFT);
            win->keys &= ~((keycode == 65) << KEY_BIT_SPACE);
            win->keys &= ~((keycode == 37) << KEY_BIT_LEFT_CTRL);
            xkb_state_update_key(keyboard->state, keycode, XKB_KEY_UP);
            break;
        }
	case XCB_DESTROY_NOTIFY:
        {
            auto destroy = (xcb_destroy_notify_event_t*)event;
            ctx[destroy->event - beginID].open = false;
            global_print("s", "XCB_DESTROY_NOTIFY\n");
		    break;
        }
	case XCB_CONFIGURE_NOTIFY:
        {
            auto cfgEvent = (xcb_configure_notify_event_t*)event;
            auto win = &ctx[cfgEvent->event - beginID];

            if((cfgEvent->width != 0 && cfgEvent->height != 0) && (win->width != cfgEvent->width || win->height != cfgEvent->height)) {
                win->width = cfgEvent->width;
                win->height = cfgEvent->height;
            }
            else {ASSERT(cfgEvent->width != 0 && cfgEvent->height != 0);}
            global_print("s", "XCB_CONFIGURE_NOTIFY\n");
            break;
        }
    case XCB_EXPOSE:
        {
            auto expose = (xcb_expose_event_t*)event;
            global_print("suc", "XCB_EXPOSE ", expose->window, '\n');
            break;
        }
    case XCB_PROPERTY_NOTIFY:
        {
            auto notify = (xcb_property_notify_event_t*)event;
            auto win =  &ctx[notify->window - beginID];
            break;
        }
    case XCB_ENTER_NOTIFY:
        break;
    case XCB_LEAVE_NOTIFY:
        break;
	default:
		break;
	}
}
void consume_xcb_events(xcb_connection_t* connection, xkb_keyboard* xkb, u32 ctxCount, xcb_context* ctx) {

    while(auto e = xcb_poll_for_event(connection)) {

        handle_xcb_event(connection, xkb, ctxCount, ctx, e);
        xcb_flush(connection);
        free(e);
    }
}
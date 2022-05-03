#include <common.h>
#include <window.h>


i32 main() {

    init_global_state(0, 0, 1024);
    xcb_connection_t* c = xcb_connect(0, 0);
    auto id0 = xcb_generate_id(c);
    auto id1 = xcb_generate_id(c);

    xcb_context ctx[2];
    ctx[0] = make_xcb_context(c, id0, 640, 480, "t0");
    ctx[1] = make_xcb_context(c, id1, 640, 480, "t1");

    for(;ctx[0].open || ctx[1].open;) {

        consume_xcb_events(c, 2, ctx);
        global_io_flush();
    }

    return 0;
}
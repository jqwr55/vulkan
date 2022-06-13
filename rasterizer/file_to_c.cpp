#include <common.h>


i32 main(i32 argc, char* argv[]) {

    auto mem = init_global_state(0, Megabyte(128), 1024);
    auto alloc = make_linear_allocator(mem, Megabyte(128));

    if(argc < 2) return 0;

    for(u32 i = 1; i < argc; i++) {

        auto top = alloc.top;
        auto file = (byte*)linear_allocator_top(&alloc);
        auto fileSize = ReadFile(argv[i], &alloc);

        i32 len = Max((i32)0, (i32)str_len(argv[i]) - 1);

        const char* fileName = nullptr;
        for(i32 k = len; k > 0; k--) {

            if(argv[i][k] == '.') {
                argv[i][k] = '_';
            }
            if(argv[i][k] == '/') {
                fileName = argv[i] + (k + 1);
                break;
            }
        }

        global_print("sss",
            "extern const byte ", fileName, "[] = {\n"
        );

        for(u32 k = 0; k < (fileSize / 8) * 8; k += 8) {
            global_print("c xs xs xs xs  xs xs xs xs",
                '\t',
                file[k + 0], ",\t", file[k + 1], ",\t", file[k + 2], ",\t", file[k + 3], ",\t", 
                file[k + 4], ",\t", file[k + 5], ",\t", file[k + 6], ",\t", file[k + 7], ",\n"
            );
        }
        file += ((fileSize / 8) * 8);

        if(fileSize % 8) {
            global_print("c", '\t');
        }
        for(u32 k = 0; k < fileSize % 8; k++) {
            global_print("xs",
                file[k], ",\t"
            );
        }
        if(fileSize % 8) {
            global_print("c", '\n');
        }

        global_print("s", "};\n");
        global_io_flush();

        alloc.top = top;
    }

    return 0;
}

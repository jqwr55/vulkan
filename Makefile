include make_includes/makefile_map
include make_includes/makefile_colors

define UNIQUE =
	$(eval seen :=)
	$(foreach _,$1,$(if $(filter $_,${seen}),,$(eval seen += $_)))
	${seen}
endef

MAKE_FILE = ./Makefile

CXX_COMPILER = g++
CXX_CFLAGS 	 = -fno-exceptions -std=c++17 -g -masm=intel
CXX_LDFLAGS  = -lvulkan -ldl -pthread -lX11 `pkg-config --cflags --libs xcb xkbcommon xkbcommon-x11`

SPV_COMPILER = glslc
SPV_CFLAGS 	 = -fshader-stage=

CXX_INCLUDE_FLAGS = -I ./src -I ./libs/include/
CXX_SRC_PATH 	  = ./src
CXX_SRC_LIB_PATH  = ./libs
BIN_PATH 		  = ./bin
GLSL_SOURCE_PATH  = ./res
EXECUTABLE_BINARY = vk_main

CXX_LDFILES 		= $(BIN_PATH)/triangulate.a
CXX_SRC_FILES 		= $(call UNIQUE, $(sort $(subst $(CXX_SRC_PATH)/,, $(wildcard $(CXX_SRC_PATH)/*.cpp)) embedded_blobs.cpp ))
CXX_OBJ_FILES 		= $(subst .cpp,.o,$(CXX_SRC_FILES))
HXX_FILES 			= $(subst $(CXX_SRC_PATH)/,, $(wildcard $(CXX_SRC_PATH)/*.h))
GLSL_FRAG_SRC_FILES = $(subst $(GLSL_SOURCE_PATH)/,, $(wildcard $(GLSL_SOURCE_PATH)/*.glsl.frag ))
GLSL_VERT_SRC_FILES = $(subst $(GLSL_SOURCE_PATH)/,, $(wildcard $(GLSL_SOURCE_PATH)/*.glsl.vert ))
SPV_FRAG_FILES 		= $(foreach element, $(GLSL_FRAG_SRC_FILES), $(subst .glsl.frag,.spv,$(element)))
SPV_VERT_FILES 		= $(foreach element, $(GLSL_VERT_SRC_FILES), $(subst .glsl.vert,.spv,$(element)))

CXX_OBJ_FILE_PATHS 	 = $(foreach element, $(CXX_SRC_FILES), $(BIN_PATH)/$(strip $(subst .cpp,.o,$(element)) ) )
FRAG_GLSL_FILE_PATHS = $(foreach element, $(GLSL_FRAG_SRC_FILES), $(GLSL_SOURCE_PATH)/$(element) )
VERT_GLSL_FILE_PATHS = $(foreach element, $(GLSL_VERT_SRC_FILES), $(GLSL_SOURCE_PATH)/$(element) )
SPV_FRAG_FILE_PATHS  = $(foreach element, $(SPV_FRAG_FILES), $(BIN_PATH)/$(element) )
SPV_VERT_FILE_PATHS  = $(foreach element, $(SPV_VERT_FILES), $(BIN_PATH)/$(element) )

CXX_LIB_PATHS 			= $(filter-out $(CXX_SRC_LIB_PATH)/include, $(wildcard $(CXX_SRC_LIB_PATH)/*))
CXX_LIB_NAMES 			= $(subst $(CXX_SRC_LIB_PATH)/,, $(CXX_LIB_PATHS))
CXX_LIB_A_FILES			= $(addsuffix .a,$(CXX_LIB_NAMES))
CXX_LIB_A_FILE_PATHS 	= $(foreach element, $(CXX_LIB_A_FILES), $(BIN_PATH)/$(element) )


strip_cpp_postfix = $(subst .cpp,,$(1))
change_cpp_to_o_postfix = $(subst .cpp,.o,$(1))

$(foreach library, $(CXX_LIB_NAMES), \
	$(eval $(library)_cxx_file_paths := $(wildcard $(CXX_SRC_LIB_PATH)/$(library)/*.cpp) ) \
)
$(foreach library, $(CXX_LIB_NAMES), \
	$(eval $(library)_hxx_file_paths := $(wildcard $(CXX_SRC_LIB_PATH)/$(library)/*.h) ) \
)
$(foreach library, $(CXX_LIB_NAMES), \
	$(eval $(library)_obj_file_paths := $(subst $(CXX_SRC_LIB_PATH)/$(library)/,$(BIN_PATH)/, $(call change_cpp_to_o_postfix, $($(library)_cxx_file_paths))) ) \
)

# $1 src file
# $2 out file
# $3 src path
# $4 bin path
# $5 compiler
# $6 compiler flags
# $(binary_to_be_produced): $(source_file)
#	$(compiler) $(flags) $(path_to_src)/$(source_file) -o $(path_to_binary_to_be_produced)/$(binary_to_be_produced)
#	ex: g++ -fno-exceptions -std=c++17 -g ./src/main.cpp -o ./bin/main.o
define make-target-compile-file
$4/$(strip $2): $3/$1 $(MAKE_FILE)
	@printf "$(GREEN)compiling $1 $(NONE)\n"
	$5 $6 $3/$1 -o $4/$(strip $2)
endef


# $1 library name
# $2 bin path
define make-target-build_lib
build_$(strip $1) : $(strip $2)/$(strip $1).a $(MAKE_FILE)
	$(info $1.a built)
endef

# $1 library name
# $2 bin path
define make-target-lib_a
$(strip $2)/$(strip $1).a : $($(strip $1_obj_file_paths)) $(MAKE_FILE)
	@printf "$(GREEN)archiving $1.a $(NONE)\n"
	ar rcs $(strip $2)/$(strip $1).a  $($(strip $1_obj_file_paths))
endef

# $1 src path
# $2 out path
# $3 compiler
# $4 compiler flags
define make-target-compile-path
$(strip $2) : $(strip $1) $(MAKE_FILE)
	@printf "$(GREEN) compiling $1 $(NONE)\n"
	$3 $(strip $1) $4 -o $(strip $2)
endef

run: $(BIN_PATH)/$(EXECUTABLE_BINARY) $(SPV_FRAG_FILE_PATHS) $(SPV_VERT_FILE_PATHS)
	clear
	cd ./bin && ./vk_main

build: $(BIN_PATH)/$(EXECUTABLE_BINARY) $(SPV_FRAG_FILE_PATHS) $(SPV_VERT_FILE_PATHS) $(CXX_LDFILES)
	@printf "$(GREEN) $(EXECUTABLE_BINARY) linked $(NONE)\n"

build_all_libs: $(CXX_LIB_A_FILE_PATHS) $(MAKE_FILE) $(CXX_LDFILES)
	@printf "$(GREEN) $(CXX_LIB_A_FILES) built $(NONE)\n"

$(BIN_PATH)/$(EXECUTABLE_BINARY): $(CXX_OBJ_FILE_PATHS) $(MAKE_FILE) $(CXX_LDFILES)
	@printf "$(GREEN) linking $(EXECUTABLE_BINARY) $(NONE)\n"
	$(CXX_COMPILER) $(CXX_CFLAGS) -o $(BIN_PATH)/$(EXECUTABLE_BINARY) $(CXX_OBJ_FILE_PATHS) $(CXX_LDFILES) $(CXX_LDFLAGS)

link_exe: $(CXX_OBJ_FILE_PATHS) $(MAKE_FILE) $(CXX_LDFILES)
	@printf "$(GREEN) linking $(EXECUTABLE_BINARY) $(NONE)\n"
	$(CXX_COMPILER) $(CXX_CFLAGS) -o $(BIN_PATH)/$(EXECUTABLE_BINARY) $(CXX_OBJ_FILE_PATHS) $(CXX_LDFILES) $(CXX_LDFLAGS)

build_cxx_obj: $(CXX_OBJ_FILE_PATHS)
	$(info compiled CXX files)

build_frag_shaders: $(SPV_FRAG_FILE_PATHS)
	$(info frag shaders compiled)

build_vert_shaders: $(SPV_VERT_FILE_PATHS)
	$(info vert shaders compiled)

purge:
	rm ./bin/*

RASTERIZER_OBJ =  ./bin/common.o ./bin/open_type_loader.o ./bin/math3d.o
RASTERIZER_A =  ./bin/triangulate.a
build_raster: ./rasterizer/main $(MAKE_FILE)

./rasterizer/main: ./rasterizer/main.cpp $(RASTERIZER_OBJ) $(RASTERIZER_A) $(MAKE_FILE)
	g++ ./rasterizer/main.cpp $(RASTERIZER_OBJ) $(RASTERIZER_A) -I ./libs/include -I ./src/ -o ./rasterizer/main -fno-exceptions -std=c++17 -g -ldl -pthread -lX11 `pkg-config --libs glfw3 gl glew`

./rasterizer/file_to_c: ./rasterizer/file_to_c.cpp
	g++ ./rasterizer/file_to_c.cpp ./src/common.cpp -I ./src/ -o ./rasterizer/file_to_c

./src/embedded_blobs.cpp: ./rasterizer/file_to_c $(SPV_FRAG_FILE_PATHS) $(SPV_VERT_FILE_PATHS) $(MAKE_FILE)
	printf "#include <common.h>\n" > ./src/embedded_blobs.cpp
	./rasterizer/file_to_c ./bin/flat.spv ./bin/vertex2d.spv >> ./src/embedded_blobs.cpp

# ./src/*.cpp -> ./bin/*.o
$(foreach element, $(CXX_SRC_FILES), \
	$(eval $(call make-target-compile-file,$(element), $(subst .cpp,.o,$(element)), $(CXX_SRC_PATH), $(BIN_PATH), $(CXX_COMPILER), $(CXX_CFLAGS) -c $(CXX_INCLUDE_FLAGS) )) \
)

# .glsl.frag -> .spv
$(foreach element, $(GLSL_FRAG_SRC_FILES), \
	$(eval $(call make-target-compile-file,$(element), $(subst .glsl.frag,.spv,$(element)), $(GLSL_SOURCE_PATH), $(BIN_PATH), $(SPV_COMPILER), $(SPV_CFLAGS)frag )) \
)

# .glsl.vert -> .spv
$(foreach element, $(GLSL_VERT_SRC_FILES), \
	$(eval $(call make-target-compile-file,$(element), $(subst .glsl.vert,.spv,$(element)), $(GLSL_SOURCE_PATH), $(BIN_PATH), $(SPV_COMPILER), $(SPV_CFLAGS)vert )) \
)
$(foreach i, $(shell seq $(words $(CXX_LIB_NAMES))), 										\
	$(foreach k, $(shell seq $(words $($(word $(i), $(CXX_LIB_NAMES))_cxx_file_paths))), 	\
		$(eval $(call make-target-compile-path, 											\
			$(word $(k), $($(word $(i), $(CXX_LIB_NAMES))_cxx_file_paths)), 				\
			$(word $(k), $($(word $(i), $(CXX_LIB_NAMES))_obj_file_paths)), 				\
			$(CXX_COMPILER), $(CXX_CFLAGS) -c -I $(word $(i), $(CXX_LIB_PATHS))/,			\
		)) 																					\
	) 																						\
	$(eval $(call make-target-lib_a,														\
		$(word $(i), $(CXX_LIB_NAMES)), $(BIN_PATH) 										\
	))																						\
	$(eval $(call make-target-build_lib,													\
		$(word $(i), $(CXX_LIB_NAMES)), $(BIN_PATH) 										\
	))																						\
)
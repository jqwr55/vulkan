COMPILER = g++
CFLAGS = -fno-exceptions -std=c++17 -g
LDFLAGS = -lvulkan -ldl -pthread -lX11 `pkg-config --cflags --libs xcb xkbcommon xkbcommon-x11`

INCLUDE_FLAGS = -I ./src -I ./libs/include/
SRC_PATH = ./src
BIN_PATH = ./bin
SHADER_SOURCE_PATH = ./res

LDFILES =
SRC_FILES = $(wildcard $(SRC_PATH)/*.cpp)
OBJ_FILES = $(subst .cpp,.o, $(subst $(SRC_PATH),$(BIN_PATH),$(SRC_FILES)))
HEADER_FILES = $(wildcard $(SRC_PATH)/*.h)

define make-target
$(subst $(SRC_PATH),$(BIN_PATH), $(subst .cpp,.o,$1)): $1
	$(COMPILER) $(CFLAGS) -c $(INCLUDE_FLAGS) $1 -o $(subst $(SRC_PATH),$(BIN_PATH), $(subst .cpp,.o,$1))
endef

run: ./bin/vk_main $(BIN_PATH)/frag_textured.spv $(BIN_PATH)/vertex3d.spv $(BIN_PATH)/frag_flat.spv $(BIN_PATH)/vertex2d.spv
	clear
	cd ./bin && ./vk_main

build: ./bin/vk_main $(BIN_PATH)/frag_textured.spv $(BIN_PATH)/vertex3d.spv $(BIN_PATH)/frag_flat.spv $(BIN_PATH)/vertex2d.spv
	$(info ready to run)

./bin/vk_main: $(OBJ_FILES)
	$(COMPILER) $(CFLAGS) -o $(BIN_PATH)/vk_main $(OBJ_FILES) $(LDFILES) $(LDFLAGS)
	
$(BIN_PATH)/frag_textured.spv: $(SHADER_SOURCE_PATH)/frag_textured.glsl
	glslc -fshader-stage=frag $(SHADER_SOURCE_PATH)/frag_textured.glsl -o $(BIN_PATH)/frag_textured.spv
$(BIN_PATH)/vertex3d.spv: $(SHADER_SOURCE_PATH)/vertex3d.glsl
	glslc -fshader-stage=vert $(SHADER_SOURCE_PATH)/vertex3d.glsl -o $(BIN_PATH)/vertex3d.spv

$(BIN_PATH)/frag_flat.spv: $(SHADER_SOURCE_PATH)/frag_flat.glsl
	glslc -fshader-stage=frag $(SHADER_SOURCE_PATH)/frag_flat.glsl -o $(BIN_PATH)/frag_flat.spv
$(BIN_PATH)/vertex2d.spv: $(SHADER_SOURCE_PATH)/vertex2d.glsl
	glslc -fshader-stage=vert $(SHADER_SOURCE_PATH)/vertex2d.glsl -o $(BIN_PATH)/vertex2d.spv

$(foreach element, $(SRC_FILES), $(eval $(call make-target,$(element))) )


COMPILER = g++
CFLAGS = -fno-exceptions -std=c++17 -g
LDFLAGS = -lvulkan -ldl -pthread -lX11

INCLUDE_FLAGS = -I ./src -I ./libs/include/
SRC_PATH = ./src
BIN_PATH = ./bin
SHADER_SOURCE_PATH = ./res

LDFILES = ./libs/libglfw3.a
SRC_FILES = $(wildcard $(SRC_PATH)/*.cpp)
OBJ_FILES = $(subst .cpp,.o, $(subst $(SRC_PATH),$(BIN_PATH),$(SRC_FILES)))
HEADER_FILES = $(wildcard $(SRC_PATH)/*.h)

define make-target
$(subst $(SRC_PATH),$(BIN_PATH), $(subst .cpp,.o,$1)): $1
	$(COMPILER) $(CFLAGS) -c $(INCLUDE_FLAGS) $1 -o $(subst $(SRC_PATH),$(BIN_PATH), $(subst .cpp,.o,$1))
endef

run: ./bin/vk_main $(BIN_PATH)/frag.spv $(BIN_PATH)/vertex.spv
	clear
	cd ./bin && ./vk_main

perf: ./bin/vk_main $(BIN_PATH)/frag.spv $(BIN_PATH)/vertex.spv
	clear
	cd ./bin
	perf stat -e branches,branch-misses,cache-misses,cache-references,cycles,instructions,idle-cycles-backend,idle-cycles-frontend,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-prefetch-misses,L1-dcache-prefetches,L1-icache-load-misses,L1-icache-loads,L1-icache-prefetches,LLC-load-misses,LLC-loads,LLC-stores,branch-load-misses,branch-loads,dTLB-load-misses,dTLB-loads,iTLB-loads-misses,iTLB-loads ./vk_main

build: ./bin/vk_main $(BIN_PATH)/frag.spv $(BIN_PATH)/vertex.spv
	$(info ready to run)

./bin/vk_main: $(OBJ_FILES)
	$(COMPILER) $(CFLAGS) -o $(BIN_PATH)/vk_main $(OBJ_FILES) $(LDFILES) $(LDFLAGS)
	
$(BIN_PATH)/frag.spv: $(SHADER_SOURCE_PATH)/frag.glsl
	glslc -fshader-stage=frag $(SHADER_SOURCE_PATH)/frag.glsl -o $(BIN_PATH)/frag.spv
$(BIN_PATH)/vertex.spv: $(SHADER_SOURCE_PATH)/vertex.glsl
	glslc -fshader-stage=vert $(SHADER_SOURCE_PATH)/vertex.glsl -o $(BIN_PATH)/vertex.spv

$(foreach element, $(SRC_FILES), $(eval $(call make-target,$(element))))

purge:
	rm ./bin/*
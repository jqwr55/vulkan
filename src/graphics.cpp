#include <graphics.h>


void ComputeCameraVelocity(Camera* cam, u8 keys, f32 speed) {

    u8 w = ((keys >> 0) & 1);
    u8 a = ((keys >> 1) & 1);
    u8 s = ((keys >> 2) & 1);
    u8 d = ((keys >> 3) & 1);
    u8 space = ((keys >> 4) & 1);
    u8 shift = ((keys >> 5) & 1);
   
    vec<f32, 2> horizontalForwardDir{cam->direction.x ,cam->direction.z};
    horizontalForwardDir = normalize(horizontalForwardDir);
    vec<f32, 2> horizontalOrtoDir{horizontalForwardDir.y , -horizontalForwardDir.x};

    i8 forward = w-s;
    i8 ortogonal = a-d;
    i8 vertical = space-shift;

    cam->vel.y += vertical * speed;
    cam->vel.x += (horizontalForwardDir.x * forward * speed) + (horizontalOrtoDir.x * ortogonal * speed);
    cam->vel.z += (horizontalForwardDir.y * forward * speed) + (horizontalOrtoDir.y * ortogonal * speed);
}

void RotateCamera(Camera* cam , f32 vertRotAngle , f32 horizRotAngle) {

    f32 cosHoriz = cos(horizRotAngle);
    f32 sinHoriz = sin(horizRotAngle);

    f32 cosVert = cos(vertRotAngle);
    f32 sinVert = sin(vertRotAngle);

    cam->direction.x = cam->direction.x * cosHoriz - cam->direction.z * sinHoriz;
    cam->direction.z = cam->direction.x * sinHoriz + cam->direction.z * cosHoriz;

    cam->direction = normalize(cam->direction);
    vec<f32, 3> right = normalize(cross(cam->direction, {0,1,0}));
    vec<f32, 3> w = normalize(cross(right, cam->direction));

    cam->direction = cam->direction * cosVert + w * sinVert;
    cam->direction = normalize(cam->direction);
}

Mat4<f32> LookAt(vec<f32, 3> from, vec<f32, 3> to, vec<f32, 3> worldUp) {

    vec<f32, 3> forward{ normalize(to - from) };
    vec<f32, 3> right{ normalize(cross(forward, worldUp)) };
    vec<f32, 3> up{ normalize(cross(right, forward)) };

    return Mat4<f32> {
        right.x , up.x , -forward.x , 0,
        right.y , up.y , -forward.y , 0,
        right.z , up.z , -forward.z , 0,
        -dot(right, from), -dot(up, from), dot(forward, from) , 1
    };
}

Mat<f32, 3,3> OrientTo(vec<f32, 3> from, vec<f32, 3> to, vec<f32, 3> worldUp) {

    vec<f32, 3> m_Z = normalize(to - from);
    vec<f32, 3> m_X = normalize(cross(worldUp, m_Z));
    vec<f32, 3> m_Y = normalize(cross(m_Z, m_X));

    return Mat<f32, 3,3> {
        m_X.x, m_X.y, m_X.z,
        m_Y.x, m_Y.y, m_Y.z,
        m_Z.x, m_Z.y, m_Z.z,
    };
}

Mat4<f32> ComputePerspectiveMat4(f32 fov , f32 aspect , f32 near , f32 far) {

    f32 tanFov = tan( fov * 0.5 );
    f32 x = 1 / ( aspect * tanFov );
    f32 y = 1 / ( tanFov );

    f32 z = -(far + near) / (far - near);
    f32 w = (-2 * far * near) / (far - near);

    return Mat4<f32> {
        x,0,0,0,
        0,-y,0,0,
        0,0,z,-1,
        0,0,w,0
    };
}
Mat<f32, 3,3> ComputeRotarionXMat4(f32 x) {

    Mat<f32, 3, 3> rot {
        1,0,0,
        0,cos(x),-sin(x),
        0,sin(x),cos(x),
    };
    return rot;
}
Mat<f32, 3,3> ComputeRotarionYMat4(f32 x) {

    Mat<f32, 3,3> rot {
        cos(x),0,sin(x),
        0,1,0,
        -sin(x),0,cos(x),
    };
    return rot;
}
Mat<f32, 3,3> ComputeRotarionZMat4(f32 x) {

    Mat<f32, 3,3> rot {
        cos(x),-sin(x),0,
        sin(x),cos(x),0,
        0,0,1,
    };
    return rot;
}

GpuHeap make_gpu_heap(void* used_blocks, u32 maxAllocCount, u32 size) {

    GpuHeap heap{};
    heap.max_block_count = maxAllocCount;
    heap.used_blocks = (GpuMemoryBlock*)used_blocks;
    heap.used_block_count = 1;
    heap.used_blocks[0].free = true;
    heap.used_blocks[0].left = ~u32(0);
    heap.used_blocks[0].right = ~u32(0);
    heap.used_blocks[0].offset = 0;
    heap.used_blocks[0].size = size;
    return heap;
}

GpuMemoryBlock* search_free_gpu_block(GpuMemoryBlock* blocks, u32 count, u32 size, u32 alignment) {

    for(u32 i = 0; i < count; i++) {
        if(blocks[i].free && (i32)(alignment - blocks[i].offset % alignment) <= (i32)(blocks[i].size - size)) {
            return blocks + i;
        }
    }
    return nullptr;
}

GpuMemoryBlock* get_unused_block(GpuHeap* heap) {

    for(u32 i = 0; i < heap->used_block_count; i++) {
        if(heap->used_blocks[i].size == 0) {
            heap->used_blocks[i].left = ~u32(0);
            heap->used_blocks[i].right = ~u32(0);
            return heap->used_blocks + i;
        }
    }

    heap->used_blocks[heap->used_block_count].left = ~u32(0);
    heap->used_blocks[heap->used_block_count].right = ~u32(0);
    return heap->used_blocks + (heap->used_block_count++);
}

MemBlock allocate_gpu_block(GpuHeap* heap, u32 size, u32 alignment) {

    //auto i = binary_search_free_blocks(heap->free_block, heap->free_block_count, size);
    auto block = search_free_gpu_block(heap->used_blocks, heap->used_block_count, size, alignment);

    if(block) {

        auto aligned = align(block->offset, alignment);
        if(aligned > block->offset || block->size > size) {

            auto fresh_block = get_unused_block(heap);
            fresh_block->free = !(aligned > block->offset);

            fresh_block->size = (fresh_block->free ? block->size - size : size);
            fresh_block->offset = (fresh_block->free ? block->offset + size : aligned);

            fresh_block->left = block - heap->used_blocks;
            fresh_block->right = block->right;

            block->right = fresh_block - heap->used_blocks;
            if(fresh_block->right != ~u32(0)) {
                heap->used_blocks[fresh_block->right].left = fresh_block - heap->used_blocks;
            }

            if(block->size + block->offset > fresh_block->size + fresh_block->offset) {

                auto end_block = get_unused_block(heap);
                end_block->free = true;
                end_block->offset = size + aligned;
                end_block->size = block->size + block->offset - end_block->offset;
                end_block->left = fresh_block - heap->used_blocks;
                end_block->right = fresh_block->right;

                fresh_block->right = end_block - heap->used_blocks;
                if(end_block->right != ~u32(0)) {
                    heap->used_blocks[end_block->right].left = end_block - heap->used_blocks;
                }
            }

            block->free = !fresh_block->free;
            block->size = (fresh_block->free ? size : aligned - block->offset);

            auto ret = (fresh_block->free ? block - heap->used_blocks : fresh_block - heap->used_blocks);
            auto retBlock = (fresh_block->free ? MemBlock{block->offset, block->size} : MemBlock{fresh_block->offset, fresh_block->size});
            retBlock.index = ret;
            return retBlock;
        }
    }

    ASSERT(false);
    return {~u64(0),~u32(0),~u32(0)};
}
void free_gpu_block(GpuHeap* heap, MemBlock mem) {

    auto block = &heap->used_blocks[mem.index];
    GpuMemoryBlock* left = block->left == ~u32(0) ? nullptr : heap->used_blocks + block->left;
    GpuMemoryBlock* right = block->right == ~u32(0) ? nullptr : heap->used_blocks + block->right;
    
    block->free = true;
    if(left) {
        if(left->free) {
            left->size += block->size;
            left->right = block->right;

            if(block->right != ~u32(0)) {
                heap->used_blocks[block->right].left = block->left;
            }

            *block = {};
            block = left;
        }
    }
    if(right) {
        if(right->free) {
            block->size += right->size;
            block->right = right->right;

            if(right->right != ~u32(0)) {
                heap->used_blocks[right->right].left = right->left;
            }
            *right = {};
        }
    }

}
void enumarate_blocks(GpuHeap* heap) {

    u32 total = 0;
    u32 totalBlockCount = 0;
    u32 allocatedTotal = 0;
    u32 allocatedTotalBlockCount = 0;
    u32 freeTotal = 0;
    u32 freeBlockCount = 0;
    u32 fragmented = 0;
    u32 maxFreeBlock = 0;

    GpuMemoryBlock* block = nullptr;
    for(u32 i = 0; i < heap->used_block_count; i++) {
        if(heap->used_blocks[i].size != 0) {
            block = heap->used_blocks + i;
            break;
        }
    }
    while(block->left != ~u32(0)) {
        block = heap->used_blocks + block->left;
    }

    do {

        u32 block_size = block->size;
        total += block_size;
        totalBlockCount++;
        if(block->free) {
            fragmented += block_size;
            freeTotal += block_size;
            freeBlockCount++;
            maxFreeBlock = Max(maxFreeBlock, block_size);
        }
        else {
            allocatedTotal += block_size;
            allocatedTotalBlockCount++;
        }

        global_print("%i%c%i%c%i%c", (u64)block, ' ', (u32)block->free, ' ', block->size, '\n');

        block = block->right == ~u32(0) ? nullptr : heap->used_blocks + block->right;
    } while(block);


    global_print("si", "\ntotal: "               , total);
    global_print("si", "\ntotal block count: "   , totalBlockCount);
    global_print("si", "\nin use: "              , allocatedTotal);
    global_print("si", "\nin use block count: "  , allocatedTotalBlockCount);
    global_print("si", "\nfree: "                , freeTotal);
    global_print("si", "\nfree block count: "    , freeBlockCount);
    global_print("sfs", "\nfragmentation: "     , ((f64)(freeTotal - maxFreeBlock) / (f64)freeTotal) * 100.0, "%\n" );
}

MemBlock get_block(GpuHeap* heap, u32 i) {
    return {heap->used_blocks[i].offset, heap->used_blocks[i].size};
}
/*
u32 binary_search_free_blocks(GpuFreeBlock* blocks, u32 blockCount, u32 size) {

    u32 low = 0;
    u32 high = blockCount - 1;

    while(low <= high) {
        u32 mid = (low + high) / 2;

        if(blocks[mid].size == size || (blocks[mid].size > size && blocks[mid - 1].size < size)) {
            return mid;
        }

        bool cond = blocks[mid].size > size;
        high = cond ? mid - 1 : high;
        low  =  cond ? low : mid + 1;
    }
    return ~u32(0);
}

void insert_free_block(GpuFreeBlock* blocks, u32 blockCount, GpuFreeBlock block) {

    u32 index = blockCount - 1;
    for(u32 i = 0; i < blockCount; i++) {
        if(blocks[i].size > block.size) {
            index = i;
            break;
        }
    }

    memcpy(blocks + index + 1, blocks + index, sizeof(GpuFreeBlock) * (blockCount - index));
    blocks[index] = block;
}
*/

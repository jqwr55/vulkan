#include <common.h>
#include <cstddef>
#include <memory>


byte *global_malloc_base;
malloc_handler_t global_out_of_memory_handler;
LinearAllocator io;

u64 Kilobyte(u64 n) {
    return n * KILO_BYTE;
}
u64 Megabyte(u64 n) {
    return n * MEGA_BYTE;
}
u64 Gigabyte(u64 n) {
    return n * GIGA_BYTE;
}
u64 Terabyte(u64 n) {
    return n * TERA_BYTE;
}
u64 Milisecond(u64 n) {
    return n * MILI_SEC;
}
u64 Second(u64 n) {
    return n * SECOND;
}
u64 Minute(u64 n) {
    return n * SECOND * 60;
}
u64 Hour(u64 n) {
    return n * SECOND * 60 * 60;
}

u64 align(u64 n, u64 alignment) {
    
    auto mod = n % alignment;
    auto rem = alignment - mod;
    auto aligned = n + rem * (mod != 0);

    return aligned;
}
void* align_pointer(void* n, u64 alignment) {

    auto mod = (uintptr_t)n % alignment;
    auto rem = alignment - mod;
    auto aligned = (byte*)n + rem * (mod != 0);

    return aligned;
}
u32 get_unique(u32* arr, u32 count) {

    u32 unique[count];
    u32 uniqueCount = 0;
    for(u32 i = 0; i < count; i++) {
        bool notUniqueElement = false;
        for(u32 k = 0; k < uniqueCount; k++) {
            notUniqueElement |= (arr[i] == unique[k]);
        }       
        unique[uniqueCount] = arr[i];
        uniqueCount += !notUniqueElement;
    }

    memcpy(arr, unique, uniqueCount * sizeof(u32));
    return uniqueCount;
}

f32 f16_to_f32(f16 f) {
    
    bool sign = f.bits >> 15;

    u32 fraction = u32(f.bits) & (~u32(0) >> 22);
    u32 exponent = u32(f.bits >> 10) & (~u32(0) >> 27);

    fraction = fraction << 13;
    u32 biased = exponent - 15;
    biased = Clamp(biased + 127, (u32)255, (u32)0);

    f32_union_u32 bit_pattern;
    bit_pattern.u = (u32(sign) << 31) | (biased << 23) | (fraction);

    return bit_pattern.f;
}
f16 f32_to_f16(f32 f) {

    f32_union_u32 bit_pattern;
    bit_pattern.f = f;
    u32 fract = bit_pattern.u & (~u32(0) >> 9);
    u32 exp = (bit_pattern.u >> 23) & (~u32(0) >> 24);
    bool sign = bit_pattern.u >> 31;

    u32 biased = exp - 127;
    biased = Clamp(biased + 15, (u32)31, (u32)0);

    fract = fract >> 13;

    f16 ret;
    ret.bits = (u16(sign) << 15) | (u16(biased) << 10) | u16(fract);

    return ret;
}


LinearAllocator make_linear_allocator(void* base, u32 size) {
    LinearAllocator stack;
    stack.base = (byte*)base;
    stack.cap = size;
    stack.top = 0;
    return stack;
}
void* linear_aligned_allocate(LinearAllocator* stack, u32 size, u32 alignment) {

    auto mem = (byte*)align_pointer(stack->base + stack->top, alignment);

    auto rem = (uintptr_t)mem % alignment;
    auto rem2 = (mem - (stack->base + stack->top));

    ASSERT((uintptr_t)mem % alignment == 0 && ((mem - (stack->base + stack->top)) < alignment));

    stack->top = (mem + size) - stack->base;

    LOG_ASSERT(stack->top <= stack->cap, "stack overflow");
    return mem;
}
void* linear_allocate(LinearAllocator* stack, u32 size) {
    auto mem = stack->base + stack->top;
    stack->top += size;
    LOG_ASSERT(stack->top <= stack->cap, "stack overflow");
    return mem;
}
void* linear_allocate_reversed(LinearAllocator* stack, u32 size) {
    auto mem = stack->base - stack->top;
    stack->top += size;
    LOG_ASSERT(stack->top <= stack->cap, "stack overflow");
    return mem;
}
void linear_deallocate(LinearAllocator* stack, u32 size) {
    LOG_ASSERT(stack->top >= size, "stack underflow");
    stack->top -= size;
}
void roll_back_linear_allocator(LinearAllocator* stack, void* pos) {
    ASSERT(pos >= stack->base && pos <= (stack->base+stack->top));
    stack->top = (byte*)pos - stack->base;
}
void* linear_allocator_top(LinearAllocator* stack) {
    return stack->base + stack->top;
}
void* linear_allocator_top_reversed(LinearAllocator* stack) {
    return stack->base - stack->top;
}
u32 linear_allocator_free_size(LinearAllocator* stack) {
    return stack->cap - stack->top;
}

CoalescingLinearAllocator make_coalescing_linear_allocator(void* mem, u32 size) {

    CoalescingLinearAllocator allocator{};
    allocator.base = (byte*)mem;
    allocator.cap = size;
    return allocator;
}
void* linear_top(CoalescingLinearAllocator* ctx) {
    return ctx->base + ctx->top;
}
struct SizedMem {
    void* mem;
    u32 size;
};
SizedMem linear_max(CoalescingLinearAllocator* ctx) {
    
    SizedMem ret{linear_top(ctx), ctx->cap - ctx->top};
    auto it = ctx->freed;

    while(it) {
        ret.size = Max(ret.size, it->size);
        ret.mem = it;
        it = it->next;
    }
    return ret;
}
void* linear_alloc(CoalescingLinearAllocator* ctx, u32 size) {

    ASSERT(size >= sizeof(SizedFreeList));
   
    auto prev = (SizedFreeList*)ctx->freed;
    auto it = ctx->freed;
    while(it) {
        if(AbsDiff(size, it->size) > sizeof(SizedFreeList)) {

            SizedFreeList* next = it->next;
            if(it->size < size) {
                auto fresh_block = (SizedFreeList*)((byte*)it + size);
                fresh_block->size = size - it->size;
                fresh_block->next = it->next;
                next = fresh_block;
            }

            prev->next = next;
            return it;
        }

        prev = it;
        it = it->next;
    }

    auto block = ctx->base + ctx->top;
    ctx->top += size;
    LOG_ASSERT(ctx->top <= ctx->cap, "out of memory");

    return block;
}
void linear_free(CoalescingLinearAllocator* ctx, void* mem, u32 size) {

    ASSERT((ctx->base <= mem) && (mem <= (ctx->base + ctx->cap)));
    LOG_ASSERT(ctx->top >= size, "allocator underflow");
    auto free = (SizedFreeList*)mem;
    free->next = nullptr;
    free->size = size;

    if(!ctx->freed) {
        ctx->freed = free;
    }
    auto it = ctx->freed;

    SizedFreeList dummy;
    SizedFreeList* prev = &dummy;
    SizedFreeList* prev_prev = &dummy;

    byte* top = ctx->base + ctx->top;
    while(it) {

        if((byte*)it + it->size == (byte*)prev) {
            it->size += prev->size;
            prev_prev->next = it;
            prev = prev_prev;
        }
        if( ((byte*)it + it->size) == (ctx->base + ctx->top) ) {
            top = (byte*)it;
            ctx->freed = it->next;
        }
        if(it < free) {
            
            free->next = it;
            prev->next = free;
            it = free;
            free = nullptr;
        }

        prev_prev = prev;
        prev = it;
        it = it->next;
    }

    if(prev < free) {
        prev->next = free;
    }

    ctx->top = (top - ctx->base);
}

void* free_list_allocate(FreeList** list) {
    auto tmp = (*list)->next;
    *list = (*list)->next->next;
    return tmp;
}

void free_list_free(FreeList** list, void* memory) {
    Mem<FreeList>(memory).next = (*list)->next;
    *list = (FreeList*)memory;
}


CircularAllocator make_circular_allocator(void* mem, u32 size) {
    
    CircularAllocator ret;
    ret.base = (byte*)mem;
    ret.cap = size;
    ret.head = 0;

    return ret;
}
void* circular_allocate(CircularAllocator* alloc, u32 size) {

    ASSERT(size <= alloc->cap);
    bool wrap = (alloc->head + size) > alloc->cap;
    alloc->head += size;
    alloc->head *= !wrap;

    return alloc->base + alloc->head;
}

RingBuffer make_ring_buffer(void* mem, u32 size) {
    
    RingBuffer ret{};
    ret.base = (byte*)mem;
    ret.cap = size;

    return ret;
}

void circular_advance_write(RingBuffer* buffer, u32 size) {

    u32 head = buffer->head + size;
    buffer->head = head * (head <= buffer->cap);
}
void* circular_get_write_ptr(RingBuffer* buffer, u32 size) {

    ASSERT(size <= buffer->cap);

    u32 head = buffer->head;
    u32 tail = buffer->tail;

    u32 tailDst = (tail - head) - 1;
    u32 capDst = buffer->cap - head;
    u32 dst = tailDst > capDst ? capDst : tailDst;

    if(dst < size && tail < size) return nullptr;
    if(capDst >= size) {
        return buffer->base + head;
    }

    return buffer->base;
}
u32 circular_read_size(RingBuffer* buffer, u32 size) {

    u32 tail = buffer->tail;
    u32 head = buffer->head;

    if(head >= tail) {
        return head - tail;
    }

    if(buffer->cap - tail < size) {
        return buffer->head;
    }
    return buffer->cap - tail;
}
void* circular_get_read_ptr(RingBuffer* buffer, u32 size) {

    u32 tail = buffer->tail;
    return buffer->base + (tail * (tail + size <= buffer->cap));
}
void circular_advance_read(RingBuffer* buffer, u32 size) {

    buffer->tail += size;
    if(buffer->tail > buffer->cap) {
        buffer->tail = 0;
    }
}

_NO_INLINE
void runtime_panic(const char* file, u32 line) {

    global_io_flush();
    global_print("%s%s%s%i%\n", "runtime panic in file: ", file, " at line: ", line);
    global_io_flush();
    exit(1);
}


bool extract_free_bit(u32 mem) {
    return bool((mem >> 31) & 1);
}
void set_free_bit(MemoryBlockHeader* block) {
    block->size |= (1 << 31);
}
void clear_free_bit(MemoryBlockHeader* block) {
    block->size = block->size & MemoryBlockHeader::FREE_BIT_MASK;
}
void set_size_in_block(MemoryBlockHeader* block, u32 size) {
    u32 free_bit = extract_free_bit(block->size);
    ASSERT(!extract_free_bit(size));
    block->size = size | (free_bit << 31);
}
u32 get_size_in_block(MemoryBlockHeader* block) {
    return block->size & block->FREE_BIT_MASK;
}
MemoryBlockHeader* get_block_ptr(byte* base, u32 smallPtr) {
    return (MemoryBlockHeader*)(smallPtr ? base + smallPtr : nullptr);
}

// observes shared global mutable state
MemoryBlockHeader *search_free_block(u32 size) {

    MemoryBlockHeader *block = (MemoryBlockHeader*)(global_malloc_base+1);
    while (block) {

#ifdef DEBUG_BUILD
        if (block->left_ptr) {
            MemoryBlockHeader* left = (MemoryBlockHeader*)(global_malloc_base + block->left_ptr);
            LOG_ASSERT(left->right_ptr == (byte*)block - global_malloc_base, "internal allocator corruption");
        }
        if (block->right_ptr) {
            MemoryBlockHeader* right = (MemoryBlockHeader*)(global_malloc_base + block->right_ptr);
            LOG_ASSERT(right->left_ptr == (byte*)block - global_malloc_base, "internal allocator corruption");
        }
#endif

        if (extract_free_bit(block->size) && get_size_in_block(block) >= size) return block;
        block = get_block_ptr(global_malloc_base, block->right_ptr);
    }

    return nullptr;
}

// mutates shared global state
void init_global_malloc(void *base_, u32 size, malloc_handler_t handler) {
    
    //null pointers are reserved
    global_malloc_base = (byte*)base_;
    MemoryBlockHeader* first_block = (MemoryBlockHeader*)(global_malloc_base+1);
    first_block->left_ptr = 0;
    first_block->right_ptr = 0;
    set_size_in_block(first_block, size);
    set_free_bit(first_block);
    global_out_of_memory_handler = handler;
}
// mutates shared global state

void* global_malloc(u32 size) {

    if (!size) return nullptr;

    MemoryBlockHeader *free_block = search_free_block(size);
    if (free_block) {
        LOG_ASSERT(extract_free_bit(free_block->size), "internal corruption");
        clear_free_bit(free_block);
        if (get_size_in_block(free_block) - size > sizeof(MemoryBlockHeader)) {

            byte *free_block_end = ((byte *)(free_block + 1)) + size;
            MemoryBlockHeader *new_free_block = (MemoryBlockHeader*)free_block_end;
            *new_free_block = {};

            set_size_in_block(new_free_block, (get_size_in_block(free_block) - size) - sizeof(MemoryBlockHeader));
            set_free_bit(new_free_block);
            new_free_block->right_ptr = free_block->right_ptr;
            new_free_block->left_ptr = (byte*)free_block - global_malloc_base;
            if (free_block->right_ptr) {
                MemoryBlockHeader* right = (MemoryBlockHeader*)(global_malloc_base + free_block->right_ptr);
                right->left_ptr = (byte*)new_free_block - global_malloc_base;
            }

            free_block->right_ptr = (byte*)new_free_block - global_malloc_base;
            set_size_in_block(free_block, size);
        }
        return free_block + 1;
    }

    global_out_of_memory_handler();
    return nullptr;
}
// mutates shared global state
void global_free(void *block) {
    if (!block) return;

    MemoryBlockHeader *header = ((MemoryBlockHeader*)block) - 1;
    LOG_ASSERT(!extract_free_bit(header->size), "double free");
    set_free_bit(header);

    auto next_block = get_block_ptr(global_malloc_base, header->right_ptr);
    auto previous_block = get_block_ptr(global_malloc_base, header->left_ptr);

#ifdef DEBUG_BUILD
    if (next_block) {
        ASSERT(next_block->left_ptr == (byte*)header - global_malloc_base);
        ASSERT(get_size_in_block(next_block) != 0);
    }
    if (previous_block) {
        ASSERT(previous_block->right_ptr == (byte*)header - global_malloc_base);
        ASSERT(get_size_in_block(previous_block) != 0);
    }
#endif

    while (next_block) {
        if (!extract_free_bit(next_block->size))
            break;

        u32 header_size = get_size_in_block(header) + get_size_in_block(next_block) + sizeof(MemoryBlockHeader);
        set_size_in_block(header, header_size);
        header->right_ptr = next_block->right_ptr;
        if (header->right_ptr) {
            auto right = (MemoryBlockHeader*)(global_malloc_base + header->right_ptr);
            right->left_ptr = (byte*)header - global_malloc_base;
        }

        next_block = get_block_ptr(global_malloc_base, header->right_ptr);
    }
    while (previous_block) {
        if (!extract_free_bit(previous_block->size))
            break;

        u32 previous_block_size = get_size_in_block(previous_block) + get_size_in_block(header) + sizeof(MemoryBlockHeader);
        set_size_in_block(previous_block, previous_block_size);
        previous_block->right_ptr = header->right_ptr;

        if (previous_block->right_ptr) {

            auto right = (MemoryBlockHeader*)(global_malloc_base + previous_block->right_ptr);
            right->left_ptr = (byte*)previous_block - global_malloc_base;
        }

        header = previous_block;
        previous_block = get_block_ptr(global_malloc_base, previous_block->left_ptr);
    }
}

// observes shared global state
void print_heap_info() {
    MemoryBlockHeader *block = (MemoryBlockHeader*)(global_malloc_base+1);
    
    u32 total = 0;
    u32 totalBlockCount = 0;
    u32 allocatedTotal = 0;
    u32 allocatedTotalBlockCount = 0;
    u32 freeTotal = 0;
    u32 freeBlockCount = 0;
    u32 fragmented = 0;
    u32 maxFreeBlock = 0;

    while (block) {

        u32 block_size = get_size_in_block(block);
        total += block_size;
        totalBlockCount++;
        if(extract_free_bit(block->size)) {
            fragmented += block_size;
            freeTotal += block_size;
            freeBlockCount++;
            maxFreeBlock = Max(maxFreeBlock, block_size);
        }
        else {
            allocatedTotal += block_size;
            allocatedTotalBlockCount++;
        }

        global_print("%i%c%i%c%i%c", (u64)block, ' ', extract_free_bit(block->size), ' ', get_size_in_block(block), '\n');
        block = get_block_ptr(global_malloc_base, block->right_ptr);
    }


    global_print("%s%i%\n", "total: "               , total);
    global_print("%s%i%\n", "total block count: "   , totalBlockCount);
    global_print("%s%i%\n", "in use: "              , allocatedTotal);
    global_print("%s%i%\n", "in use block count: "  , allocatedTotalBlockCount);
    global_print("%s%i%\n", "free: "                , freeTotal);
    global_print("%s%i%\n", "free block count: "    , freeBlockCount);
    global_print("%s%f%c%\n", "fragmentation: "     , ((f64)(freeTotal - maxFreeBlock) / (f64)freeTotal) * 100.0, '%' );
}
// observes shared global state
u32 check_live_mem(void *block) {

    if (!block)
        return ~u32(0);
    MemoryBlockHeader *header = ((MemoryBlockHeader*)block) - 1;
    ASSERT(!extract_free_bit(header->size));

    MemoryBlockHeader *next_block = get_block_ptr(global_malloc_base, header->right_ptr);
    MemoryBlockHeader *previous_block = get_block_ptr(global_malloc_base, header->left_ptr);

    if (next_block) {
        ASSERT(next_block->left_ptr == (byte*)header - global_malloc_base);
        ASSERT(get_size_in_block(next_block) != 0);
    }
    if (previous_block) {
        ASSERT(previous_block->right_ptr == (byte*)header - global_malloc_base);
        ASSERT(get_size_in_block(previous_block) != 0);
    }

    return header->size;
}
// observes shared global state
bool check_memory_integrity(void *mem) {

    if (!mem)
        return true;
    u32 size = *((u32 *)((byte *)mem - 64));
    byte *back_guard = ((byte *)mem) - 60;
    byte *front_guard = ((byte *)mem) + size;

    ASSERT((check_live_mem(back_guard - 4) - (size + 128)) <= sizeof(MemoryBlockHeader));

    bool corrupt = false;
    for (u32 i = 0; i < 60; i++) {
        corrupt |= back_guard[i] != 255;
    }
    for (u32 i = 0; i < 64; i++) {
        corrupt |= front_guard[i] != 255;
    }

    if (corrupt) {

        global_print("%\n");
        for (u32 i = 0; i < 60; i++) {
            global_print("%i%c", (u32)back_guard[i], ' ');
        }
        global_print("%\n");
        for (u32 i = 0; i < 64; i++) {
            global_print("%i%c", (u32)front_guard[i], ' ');
        }
        global_print("%s%\n", "heap corruption detected");
    }

    ASSERT(!corrupt);
    return corrupt;
}
// observes shared global state
void check_all_memory(void *check) {
    if (check != nullptr) {
        ASSERT(!check_memory_integrity(check));
    }

    bool found = false;
    const MemoryBlockHeader *block = (MemoryBlockHeader*)(global_malloc_base+1);
    while (block) {
        byte *mem = (byte *)block;
        if (!extract_free_bit(block->size)) {
            check_memory_integrity(mem + 64 + sizeof(MemoryBlockHeader));
        }
        if (check != nullptr && check == (mem + 64 + sizeof(MemoryBlockHeader))) {
            found = true;
        }
        block = get_block_ptr(global_malloc_base, block->right_ptr);
    }

    if (check != nullptr) {
        ASSERT(found);
    }
}
// mutates shared global state
void *global_malloc_debug(u32 size) {

#ifdef DEBUG_BUILD
    byte *mem = (byte *)global_malloc(size + 128);
    Mem<u32>(mem) = size;
    memset(mem + sizeof(u32), 255, 60 + 64 + size);
    check_all_memory(mem + 64);
    ASSERT((check_live_mem(mem) - (size + 128)) <= sizeof(MemoryBlockHeader));
    return mem + 64;
#else
    return global_malloc(size);
#endif
}
u32 get_allocation_size_debug(void* mem) {

#ifdef DEBUG_BUILD
    byte* allocation = (byte*)mem - 64;
    MemoryBlockHeader *header = ((MemoryBlockHeader*)allocation) - 1;
    return get_size_in_block(header);
#else
    MemoryBlockHeader *header = ((MemoryBlockHeader*)mem) - 1;
    return get_size_in_block(header);
#endif
}
// mutates shared global state
void global_free_debug(void *mem) {

#ifdef DEBUG_BUILD
    if (!mem)
        return;
    u32 size = *((u32 *)((byte *)mem - 64));
    byte *back_guard = ((byte *)mem) - 60;
    byte *front_guard = ((byte *)mem) + size;

    for (u32 i = 0; i < 60; i++) {
        ASSERT(back_guard[i] == 255);
    }
    for (u32 i = 0; i < 64; i++) {
        ASSERT(front_guard[i] == 255);
    }
    check_all_memory(mem);
    ASSERT((check_live_mem(back_guard - 4) - (size + 128)) <= sizeof(MemoryBlockHeader));
    global_free(back_guard - 4);
#else
    return global_free(mem);
#endif
}

byte* get_local_malloc_base(LocalMallocState state) {
    return ((byte*)state.headBlock) - 1;
}
LocalMallocState make_local_malloc(byte* base, u32 size) {

    LocalMallocState state;
    state.headBlock = (MemoryBlockHeader*)(base+1);
    state.headBlock->left_ptr = 0;
    state.headBlock->right_ptr = 0;
    set_free_bit(state.headBlock);
    set_size_in_block(state.headBlock, size);
    return state;
}
MemoryBlockHeader *local_search_free_block(LocalMallocState* state, u32 size) {

    byte* base = get_local_malloc_base(*state);
    MemoryBlockHeader *block = state->headBlock;
    while (block) {

#ifdef DEBUG_BUILD
        if (block->left_ptr) {
            MemoryBlockHeader* left = (MemoryBlockHeader*)(base + block->left_ptr);
            LOG_ASSERT(left->right_ptr == (byte*)block - base, "internal allocator corruption");
        }
        if (block->right_ptr) {
            MemoryBlockHeader* right = (MemoryBlockHeader*)(base + block->right_ptr);
            LOG_ASSERT(right->left_ptr == (byte*)block - base, "internal allocator corruption");
        }
#endif

        if (extract_free_bit(block->size) && get_size_in_block(block) >= size) return block;
        block = get_block_ptr(base, block->right_ptr);
    }

    return nullptr;
}
void* local_malloc(LocalMallocState* state, u32 size) {
    
    if (!size)
        return nullptr;

    byte* base = get_local_malloc_base(*state);

    MemoryBlockHeader *free_block = local_search_free_block(state, size);
    auto free_block_size = get_size_in_block(free_block);

    if (free_block) {
        LOG_ASSERT(extract_free_bit(free_block->size), "internal corruption");
        clear_free_bit(free_block);
        if (get_size_in_block(free_block) - size > sizeof(MemoryBlockHeader)) {

            byte *free_block_end = ((byte *)(free_block + 1)) + size;
            MemoryBlockHeader *new_free_block = (MemoryBlockHeader*)free_block_end;
            *new_free_block = {};

            set_size_in_block(new_free_block, (get_size_in_block(free_block) - size) - sizeof(MemoryBlockHeader));
            set_free_bit(new_free_block);
            new_free_block->right_ptr = free_block->right_ptr;
            new_free_block->left_ptr = (byte*)free_block - base;

            if (free_block->right_ptr) {
                MemoryBlockHeader* right = (MemoryBlockHeader*)(base + free_block->right_ptr);
                right->left_ptr = (byte*)new_free_block - base;
            }         

            free_block->right_ptr = (byte*)new_free_block - base;
            set_size_in_block(free_block, size);
        }
        return free_block + 1;
    }

    ASSERT(false);
}
void local_free(LocalMallocState* state, void* block) {
    
    if (!block) return;

    byte* base = get_local_malloc_base(*state);
    MemoryBlockHeader *header = ((MemoryBlockHeader*)block) - 1;
    LOG_ASSERT(!extract_free_bit(header->size), "double free");
    set_free_bit(header);

    auto next_block = get_block_ptr(base, header->right_ptr);
    auto previous_block = get_block_ptr(base, header->left_ptr);

#ifdef DEBUG_BUILD
    if (next_block) {
        ASSERT(next_block->left_ptr == (byte*)header - base);
        ASSERT(get_size_in_block(next_block) != 0);
    }
    if (previous_block) {
        ASSERT(previous_block->right_ptr == (byte*)header - base);
        ASSERT(get_size_in_block(previous_block) != 0);
    }
#endif

    while (next_block) {
        if (!extract_free_bit(next_block->size))
            break;

        u32 header_size = get_size_in_block(header) + get_size_in_block(next_block) + sizeof(MemoryBlockHeader);
        set_size_in_block(header, header_size);
        header->right_ptr = next_block->right_ptr;
        if (header->right_ptr) {
            auto right = (MemoryBlockHeader*)(base + header->right_ptr);
            right->left_ptr = (byte*)header - base;
        }

        next_block = get_block_ptr(base, header->right_ptr);
    }
    while (previous_block) {
        if (!extract_free_bit(previous_block->size))
            break;

        u32 previous_block_size = get_size_in_block(previous_block) + get_size_in_block(header) + sizeof(MemoryBlockHeader);
        set_size_in_block(previous_block, previous_block_size);
        previous_block->right_ptr = header->right_ptr;

        if (previous_block->right_ptr) {

            auto right = (MemoryBlockHeader*)(base + previous_block->right_ptr);
            right->left_ptr = (byte*)previous_block - base;
        }

        header = previous_block;
        previous_block = get_block_ptr(base, previous_block->left_ptr);
    }
}
void local_malloc_shrink(LocalMallocState* state, void* block, u32 size) {

    LOG_ASSERT(size, "size must be > 0");
    MemoryBlockHeader* header = ((MemoryBlockHeader*)block) - 1;
    LOG_ASSERT(!extract_free_bit(header->size), "use after free");
    u32 block_size = get_size_in_block(header);
    LOG_ASSERT(size <= block_size, "block cannot grow");

    byte* base = get_local_malloc_base(*state);
    MemoryBlockHeader* right_block = get_block_ptr(base, header->right_ptr);

    MemoryBlockHeader* fresh_block = (MemoryBlockHeader*)((byte*)block + size);
    fresh_block->left_ptr = (byte*)header - base;
    fresh_block->right_ptr = right_block ? (byte*)right_block - base : 0;
    set_free_bit(fresh_block);
    set_size_in_block(fresh_block, block_size - size);

    header->right_ptr = (byte*)fresh_block - base;
    if(right_block) {
        right_block->left_ptr = (byte*)fresh_block - base;
    }

    set_size_in_block(header, size);
}
u32 local_malloc_allocation_size(void* block) {
    MemoryBlockHeader* header = ((MemoryBlockHeader*)block) - 1;
    return get_size_in_block(header);
}
void* local_max_malloc(LocalMallocState* state) {

    MemoryBlockHeader* it = state->headBlock;
    byte* base = get_local_malloc_base(*state);

    MemoryBlockHeader* max_block = it;
    u32 max_size = 0;
    while(it) {

        u32 size = get_size_in_block(it);
        bool free = extract_free_bit(it->size);
        bool cond = (size > max_size) && free;
        max_size = cond ? size : max_size;
        max_block = cond ? it : max_block;

        it = get_block_ptr(base, it->right_ptr);
    }

    clear_free_bit(max_block);
    return (void*)(max_block + 1);
}
void print_local_heap_info(LocalMallocState state) {
    MemoryBlockHeader *block = state.headBlock;
    
    u32 total = 0;
    u32 totalBlockCount = 0;
    u32 allocatedTotal = 0;
    u32 allocatedTotalBlockCount = 0;
    u32 freeTotal = 0;
    u32 freeBlockCount = 0;
    u32 fragmented = 0;
    u32 maxFreeBlock = 0;

    auto base = get_local_malloc_base(state);
    while (block) {

        u32 block_size = get_size_in_block(block);
        total += block_size;
        totalBlockCount++;
        if(extract_free_bit(block->size)) {
            fragmented += block_size;
            freeTotal += block_size;
            freeBlockCount++;
            maxFreeBlock = Max(maxFreeBlock, block_size);
        }
        else {
            allocatedTotal += block_size;
            allocatedTotalBlockCount++;
        }

        global_print("ucscuc", (u64)block, ' ', extract_free_bit(block->size) ? "free" : "used", ' ', get_size_in_block(block), '\n');
        block = get_block_ptr(base, block->right_ptr);
    }


    global_print("suc", "total: "               , total, '\n');
    global_print("suc", "total block count: "   , totalBlockCount, '\n');
    global_print("suc", "in use: "              , allocatedTotal, '\n');
    global_print("suc", "in use block count: "  , allocatedTotalBlockCount, '\n');
    global_print("suc", "free: "                , freeTotal, '\n');
    global_print("suc", "free block count: "    , freeBlockCount, '\n');
    global_print("sfs", "fragmentation: "     , ((f64)(freeTotal - maxFreeBlock) / (f64)freeTotal) * 100.0, "%\n" );
}


u32 str_hash(const char *str, u32 c) {
    u32 hash = 7;
    for (u32 i = 0; i < c; i++) {
        hash = hash * 31 + str[i];
    }
    return hash;
}

// counts sentinel
u32 str_len(const char *str) {
    const char* it = str;
    while(*it++);
    return it - str;
}
void* str_cpy(void* dst, const char* src) {

    ASSERT(dst && src);
    char* dst_str = (char*)dst;
    for(;;) {
        *dst_str = *src;
        dst_str++;
        if(! (*src++) ) break;
    }

    return dst_str;
}
bool str_cmp(const char *str0, const char *str1) {
    ASSERT(str0 != nullptr && str1 != nullptr);
    for(;;) {
        if(!*str0) break;
        if(!*str1) break;
        if(*str0++ != *str1++) return false;
    }
    
    return *str0 == *str1;
}
i64 i64_power(i64 base, i64 exp) {
    i64 ret = base;
    for(u32 i = 0; i < exp; i++) {
        ret *= base;
    }
    return ret;
}
u32 u64_to_hex_string(char *buffer, u32 buffer_size, u64 n) {
  
    // buffer too small
    if(buffer_size < 3) return 0;

    // can't divide by 0 early out
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer += 2;
    if (n == 0) {
        buffer[0] = '0';
        return 3;
    }

    i64 i = 0;
    u64 m = n;
    while (m != 0) {

        // compiler should optimize division by constant
        m /= 16;
        if(i != buffer_size) {
            // count number of digits to write upto the remaining buffer size
            i++;
        }
        else {
            // otherwise shift (n) one digit at a time
            n /= 16;
        }
    }

    const char* digits = "0123456789ABCDEF";

    u32 size = i--;
    // stop at i == 0
    for (; i > -1; i--) {
        // write digit
        buffer[i] = digits[n % 16];
        n /= 16;
    }
    return size + 2;
}
u32 u64_to_string(char *buffer, u32 buffer_size, u64 n) {

    static_assert(sizeof(char) == 1);
    // buffer too small
    if(buffer_size == 0) return 0;

    // can't divide by 0 early out
    if (n == 0) {
        buffer[0] = '0';
        return 1;
    }

    i64 i = 0;
    u64 m = n;
    while (m != 0) {

        // compiler should optimize division by constant
        m /= 10;
        if(i != buffer_size) {
            // count number of digits to write upto the remaining buffer size
            i++;
        }
        else {
            // otherwise shift (n) one digit at a time
            n /= 10;
        }
    }

    u32 size = i--;
    // stop at i == 0
    for (; i > -1; i--) {
        // write digit
        buffer[i] = (n % 10 + '0');
        n /= 10;
    }
    return size;
}
u32 f32_to_string(char* buff, u32 buff_size, f32 n, u32 precision) {
    
    // char must be 1 byte
    static_assert(sizeof(char) == 1);
    // buffer too small
    if(buff_size == 0) return 0;
    u32 buffer_offset = 0;
    if(n < 0.f) {
        // n negative
        // buffer is atleast 1 byte
        buff[buffer_offset++] = '-';
    }

    n = Abs(n);
    u64 integer_part = (u64)n;
    f32 fractional_part = n - (f32)integer_part;
    // write integer part into remaining buffer
    buffer_offset += u64_to_string(buff+buffer_offset, buff_size-buffer_offset, integer_part);
    
    if(buff_size > buffer_offset && precision != 0) {

        // write fractional part if buffer has enough space
        buff[buffer_offset++] = '.';
        fractional_part *= (f32)i64_power(10, precision);
        u32 fract_size = u64_to_string(buff+buffer_offset, buff_size-buffer_offset, (u64)fractional_part);
        buffer_offset += fract_size;
    }
    return buffer_offset;
}

// expects sentinel
f64 str_to_f64(const char* str) {

    ASSERT(str);
    u64 integer = 0;
    for(;;) {
        if(*str && *str == '.') {
            str++;
            break;
        }
        u64 digit = (u64)(*str++ - '0');
        integer = integer * 10 + digit;
    }
    u64 fract = 0;
    f64 div = 1;
    for(;*str;) {
        u64 digit = (u64)(*str++ - '0');
        fract = fract * 10 + digit;
        div *= 0.1;
    }
    return (f64)integer + ((f64)fract * div);
}


byte* print_fn_v(byte* buffer, u32 buffer_size, const char* format, va_list args) {

    if(buffer_size == 0) return buffer;

    auto buffer_end = buffer + buffer_size;
    auto buffer_begin = buffer;
    while(*format && buffer != buffer_end) {
        
        while(*format == ' ') format++;
        format += *format == ' ';

        switch(*format++) {
        case 'c':// char
            {
                char arg = va_arg(args, int);
                Mem<char>(buffer++) = arg;
                break;
            }
        case 'x':// u64 hex
            {
                u64 arg = va_arg(args, u64);
                buffer += u64_to_hex_string((char*)buffer, buffer_end - buffer, arg);
                break;
            }
        case 'i':// i64
        case 'u':// u64
            {
                u64 n;
                if(format[-1] == 'i') {
                    i64 arg = va_arg(args, i64);
                    if(arg < 0) Mem<char>(buffer++) = '-';
                    n = Abs(arg);
                }
                else {
                    n = va_arg(args, u64);
                }
                buffer += u64_to_string((char*)buffer, buffer_end - buffer, n);
                break;
            }
        case 'f': // floating-point
            {
                f64 arg = va_arg(args, f64);
                buffer += f32_to_string((char*)buffer, buffer_end - buffer, (f32)arg, 7);
                break;
            }
        case 's':// null-terminated string
            {
                char next_char = *format;
                char* arg = va_arg(args, char*);//s* sized string
                i64 len;
                
                if(next_char == '*') {
                    format++;
                    len = va_arg(args, u64);
                }
                else {
                    len = str_len(arg) - 1;
                }

                len = Min(len, buffer_end - buffer);
                ASSERT(len > -1);
                memcpy(buffer, arg, len);
                buffer += len;
                break;
            }
        }
    }
    return buffer;
}

// mutates shared global state
void init_global_print(LinearAllocator memory) {
    ASSERT(memory.base);
    ASSERT(memory.cap > 64);
    ASSERT(memory.top == 0);
    io = memory;
}
void print_out_of_memory() {
    global_print("s", "out of memory\n");
    global_io_flush();
    runtime_panic(__FILE__, __LINE__);
}
byte* init_global_state(u32 heapSize, u32 miscMemoySize, u32 ioBufferSize) {

    auto memory = mmap(nullptr, heapSize + miscMemoySize + ioBufferSize, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    if(!memory) {
        size_t size = sizeof("initial memory request failed\n");
        write(STDOUT_FILENO, "initial memory request failed\n", size);
    }

    init_global_malloc(memory, heapSize, print_out_of_memory);
    auto io_base = (byte*)memory + heapSize;
    init_global_print( make_linear_allocator((byte*)io_base, ioBufferSize) );

    return (byte*)memory + heapSize + ioBufferSize;
}
// mutates shared global state
void global_io_flush() {
    write(STDOUT_FILENO, io.base, io.top);
    roll_back_linear_allocator(&io, io.base);
}
// mutates shared global state
void global_print(const char* format ...) {

    va_list args;
    va_start(args, format);
    auto end = print_fn_v(io.base+io.top, linear_allocator_free_size(&io), format, args);
    va_end(args);

    auto top = (byte*)linear_allocator_top(&io);
    if( (end - io.base) >= io.cap) {
        global_io_flush();
        top = (byte*)linear_allocator_top(&io);

        va_start(args, format);
        end = print_fn_v(top, linear_allocator_free_size(&io), format, args);
        va_end(args);

        ASSERT(end != top);
    }
    linear_allocate(&io, end-top);
}
byte* local_print(byte* buffer, u32 buffer_size, const char* format ...) {
    va_list args;
    va_start(args, format);
    auto end = print_fn_v(buffer, buffer_size, format, args);
    va_end(args);
    return end;
}


u64 ReadFile(const char* fileName, byte* buffer) {

    FILE* file = fopen(fileName, "r");
    u64 size = ~u64(0);
    if(file) {

        fseek(file, 0, SEEK_END);
        size = ftell(file);
        fseek(file, 0, SEEK_SET);

        fread(buffer, size, 1, file);
        fclose(file);
    }

    return size;
}
u64 ReadFileTerminated(const char* fileName, byte* buffer) {

    FILE* file = fopen(fileName, "r");
    u64 size = ~u64(0);
    if(file) {

        fseek(file, 0, SEEK_END);
        size = ftell(file);
        fseek(file, 0, SEEK_SET);

        fread(buffer, size, 1, file);
        buffer[size++] = 0;
        fclose(file);
    }

    return size;
}
byte* ReadFileTerminated(const char* fileName, byte* buffer, u32* size_) {

    byte *sourceString = nullptr;
    FILE *file = fopen(fileName, "r");
    if (file) {

        fseek(file, 0, SEEK_END);
        u32 size = ftell(file);
        fseek(file, 0, SEEK_SET);
        *size_ = size + 1;

        if (!buffer) {
            sourceString = (byte *)LOG(global_malloc_debug(size + 1));
        }
        else {
            sourceString = buffer;
        }

        fread(sourceString, size, 1, file);
        sourceString[size] = 0;
        fclose(file);
    }

#ifdef DEBUG_BUILD
    check_all_memory(nullptr);
#endif

    return sourceString;
}
byte* ReadFile(const char* fileName, byte* buffer, u32* size_) {

    byte *sourceString = nullptr;
    FILE *file = fopen(fileName, "r");
    if (file) {

        fseek(file, 0, SEEK_END);
        u32 size = ftell(file);
        fseek(file, 0, SEEK_SET);
        *size_ = size + 1;

        if (!buffer) {
            sourceString = (byte *)LOG(global_malloc_debug(size));
        }
        else {
            sourceString = buffer;
        }

        fread(sourceString, size, 1, file);
        fclose(file);
    }

#ifdef DEBUG_BUILD
    check_all_memory(nullptr);
#endif

    return sourceString;
}

u32 u32_log2(u32 n) {
    return 31 - __builtin_clz(n);
}
f32 f32_log(f32 n, f32 base) {
    return (f32)u32_log2((u32)n) / (f32)u32_log2((u32)base);
}
void local_free_wrapper(LocalMallocState* state, void* mem, u32 size) {
    local_free(state, mem);
}

ImageDescriptor LoadBMP(const char* path, void* memory) {

    FILE* ImgHandle;
    ImgHandle = fopen(path, "r");
    if(!ImgHandle) {
        global_print("ssc", "file could not be opened ", path, '\n');
        return ImageDescriptor{};
    }

    fseek(ImgHandle, 0, SEEK_END);
    u32 sizeofBmp = ftell(ImgHandle);
    fseek(ImgHandle, 0, SEEK_SET);
    fread(memory + sizeofBmp, sizeofBmp, 1, ImgHandle);
    fclose(ImgHandle);

    byte* bmp = (byte*)memory + sizeofBmp;
    if (Mem<char>(bmp) != 'B' || Mem<char>(bmp + 1) != 'M') {
        global_print("s", "Not a bmp file\n");
        return ImageDescriptor{};
    }

    u32 compression = Mem<u32>(bmp + 30);
    if(compression != 0) {
        global_print("s", "Error image compressed\n");
        return ImageDescriptor{};
    }
    u32 imageOffSet = Mem<u32>(bmp + 10);
    i32 imageWidth  = Mem<i32>(bmp + 18);
    i32 imageHeight = Mem<i32>(bmp + 22);

    ImageDescriptor descriptor{};
    descriptor.img = (Pixel*)memory;
    descriptor.height = imageHeight;
    descriptor.width = imageWidth;
    u32 rowOffset = 0;

    if (imageWidth * 3 % 4 != 0) {
        rowOffset = 4 - (imageWidth * 3 % 4);
    }

    byte* img = bmp + imageOffSet;
    for (i32 i = 0; i < imageHeight; i++) {
        for (i32 k = 0; k < imageWidth; k++) {
            descriptor.img[i * imageWidth + k].b = img[(imageWidth * 3 + rowOffset) * i + k * 3 + 0];
            descriptor.img[i * imageWidth + k].g = img[(imageWidth * 3 + rowOffset) * i + k * 3 + 1];
            descriptor.img[i * imageWidth + k].r = img[(imageWidth * 3 + rowOffset) * i + k * 3 + 2];
            descriptor.img[i * imageWidth + k].a = 0;
            descriptor.img[i * imageWidth + k].a = (descriptor.img[i * imageWidth + k].mem == 0 ? 0 : 255);
        }
    }

    return descriptor;
}



bool VerifyPngSignature(u64 sig) {
    constexpr u64 PNG = 727905341920923785;
    return sig == PNG;
}
bool VerifyPngSignature(const byte* mem) {

    bool ret = true;
    u8 signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    for(u32 i = 0; i < SIZE_OF_ARRAY(signature); i++) {
        ret &= (mem[i] == signature[i]);
    }
    return ret;
}

constexpr u32 PNG_TYPE_STR(char c0, char c1, char c2, char c3) {
    return ((u32)c3 << 24) | ((u32)c2 << 16) | ((u32)c1 << 8) | (u32)c0;
}
u16 ReverseByteOrder(u16 n) {
    
    u16 b1 = 255 & (n >> 0);
    u16 b0 = 255 & (n >> 8);
    return b1 << 8 | b0;
}
u32 ReverseByteOrder(u32 n) {
    
    u32 b3 = 255 & (n >> 0);
    u32 b2 = 255 & (n >> 8);
    u32 b1 = 255 & (n >> 16);
    u32 b0 = 255 & (n >> 24);

    return b3 << 24 | b2 << 16 | b1 << 8 | b0;
}
u64 ReverseByteOrder(u64 n) {
    
    u64 b7 = 255 & (n >> 0);
    u64 b6 = 255 & (n >> 8);
    u64 b5 = 255 & (n >> 16);
    u64 b4 = 255 & (n >> 24);
    u64 b3 = 255 & (n >> 32);
    u64 b2 = 255 & (n >> 40);
    u64 b1 = 255 & (n >> 48);
    u64 b0 = 255 & (n >> 56);

    return b7 << 56 | b6 << 48 | b5 << 40 | b4 << 32 | b3 << 24 | b2 << 16 | b1 << 8 | b0;
}


void make_crc_table(u64* crc_table) {
    
    u64 c;
    for (u32 i = 0; i < 256; i++) {
        
        c = (u64)i;
        for (u32 k = 0; k < 8; k++) {
            if (c & 1) {
                c = 0xedb88320L ^ (c >> 1);
            }
            else {
                c = c >> 1;
            }
        }
        crc_table[i] = c;
    }
}

u64 compute_crc_(const u64* crc_table, u64 iv, byte* mem, u32 size) {
    
    u64 c = iv;
    for (u32 i = 0; i < size; i++) {
        c = crc_table[(c ^ mem[i]) & 0xff] ^ (c >> 8);
    }
    return c;
}
u64 compute_crc(const u64* crc_table, byte* mem, u32 size) {
    return compute_crc_(crc_table, 0xffffffffL, mem, size) ^ 0xffffffffL;
}

ChromaPayload LoadChroma(void* mem) {

    auto payload = (ChromaPayload*)mem;
    ChromaPayload ret;
    ret.whitePointX = ReverseByteOrder(payload->whitePointX);
    ret.whitePointY = ReverseByteOrder(payload->whitePointY);
    ret.redX = ReverseByteOrder(payload->redX);
    ret.redY = ReverseByteOrder(payload->redY);
    ret.greenX = ReverseByteOrder(payload->greenX);
    ret.greenY = ReverseByteOrder(payload->greenY);
    ret.blueX = ReverseByteOrder(payload->blueX);
    ret.blueY = ReverseByteOrder(payload->blueY);
    return ret;
}
bool PNGCheckColorTypeBitDepth(PNGInfo* info) {

    constexpr u8 validBitdepths0[] = {5, 1,2,4,8,16};
    constexpr u8 validBitdepths1[] = {2, 8,16};
    constexpr u8 validBitdepths2[] = {4, 1,2,4,8};
    constexpr u8 validBitdepths3[] = {0};
    const u8* validBitdepths[] = {
        validBitdepths0,
        validBitdepths3,
        validBitdepths1,
        validBitdepths2,
        validBitdepths1,
        validBitdepths3,
        validBitdepths1
    };

    auto bitdepths = validBitdepths[info->colorType];
    for(u32 i = 0; i < bitdepths[0]; i++) {

        if(info->bitDepth == bitdepths[i + 1]) {
            return true;
        }
    }
    return false;
}



void PrintPNGChunks(const byte* pngMemory, Printer printer) {

    if(!VerifyPngSignature(pngMemory)) {
        printer.print(printer.user, "s", "signature missmatch\n");
    }

    bool firstIdat = true;
    const byte* mem = pngMemory + 8;
    for(bool run = true; run ;) {

        auto chunk = (PNGChunk*)mem;
        switch(chunk->type) {
        case PNG_TYPE_STR('I', 'H', 'D', 'R'):
            {
                
                printer.print(printer.user, "xs", mem - pngMemory, " IHDR\n");

                auto ihdr = Mem<IHDRPayload>(chunk->payload);
                ihdr.height = ReverseByteOrder(ihdr.height);
                ihdr.width = ReverseByteOrder(ihdr.width);
                const char* colorTypeStrs[] = {
                    "grayscale",
                    nullptr,
                    "R,G,B triple",
                    "palette",
                    "grayscale + alpha",
                    nullptr,
                    "R,G,B,A",
                };

                printer.print(printer.user, "suc", " height: ", (u64)ihdr.height, '\n');
                printer.print(printer.user, "suc", " width: ", (u64)ihdr.width, '\n');
                printer.print(printer.user, "suc", " bitDepth: ", (u64)ihdr.bitDepth, '\n');
                printer.print(printer.user, "sucss", " colorType: ", (u64)ihdr.colorType, '(', colorTypeStrs[ihdr.colorType], ")\n");
                printer.print(printer.user, "su", " compressionMethod: ", (u64)ihdr.compressionMethod);
                if(ihdr.compressionMethod == 0) {
                    printer.print(printer.user, "s", "(deflate)\n");
                }
                printer.print(printer.user, "suc", " filterMethod: ", (u64)ihdr.filterMethod, '\n');
                printer.print(printer.user, "suc", " interlaceMethod: ", (u64)ihdr.interlaceMethod, '\n');
                break;
            }
        case PNG_TYPE_STR('I', 'D', 'A', 'T'):
            {
                printer.print(printer.user, "xs", mem - pngMemory, " IDAT\n");

                if(firstIdat) {
                    firstIdat = false;

                    auto CMF = chunk->payload[0];
                    auto FLG = chunk->payload[1];

                    u16 check = (CMF * 256 + FLG);
                    if(check % 31 != 0) {
                        printer.print(printer.user, "s", " zlib corruption: CMF*256 + FLG is not multiple of 31\n");
                    }

                    u8 CM = CMF & 0xF;
                    u8 CINFO = CMF >> 4;
                    u8 FCHECK = FLG & 0x1F;
                    u8 FDICT = (FLG >> 5) & 1;
                    u8 FLEVEL = (FLG >> 6);

                    printer.print(printer.user, "suc", " CM: ", (u64)CM, '\n');
                    printer.print(printer.user, "suc", " CINFO: ", (u64)CINFO, '\n');
                    printer.print(printer.user, "suc", " FCHECK: ", (u64)FCHECK, '\n');
                    printer.print(printer.user, "suc", " FDICT: ", (u64)FDICT, '\n');
                    printer.print(printer.user, "suc", " FLEVEL: ", (u64)FLEVEL, '\n');
                }

                break;
            }
        case PNG_TYPE_STR('I', 'E', 'N', 'D'):
            {
                printer.print(printer.user, "xs", mem - pngMemory, " IEND\n");
                run = false;
                break;
            }
        case PNG_TYPE_STR('t', 'E', 'X', 't'):
            {

                auto keyWordLen = str_len((const char*)chunk->payload);
                auto chunkS = ReverseByteOrder(chunk->length);
                auto textLen = chunkS - keyWordLen;
                printer.print(printer.user, "xss*cs*c", mem - pngMemory, " tEXt\n ", chunk->payload, keyWordLen, ' ', chunk->payload + keyWordLen, textLen, '\n');

                break;
            }
        case PNG_TYPE_STR('g', 'A', 'M', 'A'):
        case PNG_TYPE_STR('c', 'H', 'R', 'M'):
        case PNG_TYPE_STR('s', 'R', 'G', 'B'):
        case PNG_TYPE_STR('i', 'C', 'C', 'P'):
        case PNG_TYPE_STR('z', 'T', 'X', 't'):
        case PNG_TYPE_STR('i', 'T', 'X', 't'):
        case PNG_TYPE_STR('b', 'K', 'G', 'D'):
        case PNG_TYPE_STR('s', 'B', 'I', 'T'):
        case PNG_TYPE_STR('s', 'P', 'L', 'T'):
        case PNG_TYPE_STR('h', 'I', 'S', 'T'):
        case PNG_TYPE_STR('t', 'I', 'M', 'E'):
            {
                printer.print(printer.user, "xcs*c", mem - pngMemory, ' ', &chunk->type, 4, '\n');
                break;
            }
        default:
            {
                printer.print(printer.user, "xss*c", mem - pngMemory, " unkonw chunk: ", &chunk->type, 4, '\n');
                break;
            }
        }

        u32* end = (u32*)(chunk->payload + ReverseByteOrder(chunk->length));
        mem = (byte*)(end + 1);
    }

    printer.flush(printer.user);
}


struct BitStream {
    byte* bytePtr;
    u32 bitPtr;
};
u32 PeakBits(const BitStream* stream, u32 bitCount) {

    ASSERT(bitCount < 33);
    auto bitPtr = stream->bitPtr;
    auto bytePtr = stream->bytePtr;

    bytePtr += bitPtr >> 3;
    bitPtr &= 7;

    byte b = *bytePtr;
    u32 ret = 0;
    for(u32 i = 0; i < bitCount; i++) {
        
        u32 bit = (b >> bitPtr) & 1;
        ret |= (bit << i);

        bitPtr++;
        if(bitPtr == 8) {
           b = *(++bytePtr);
           bitPtr = 0;
        }
    }

    return ret;    
}
u32 ReadBits(BitStream* stream, u32 bitCount) {

    ASSERT(bitCount < 33);

    auto bitPtr = stream->bitPtr;
    auto bytePtr = stream->bytePtr;

    bytePtr += bitPtr >> 3;
    bitPtr &= 7;

    byte b = *bytePtr;
    u32 ret = 0;
    for(u32 i = 0; i < bitCount; i++) {
        
        u32 bit = (b >> bitPtr) & 1;
        ret |= (bit << i);

        bitPtr++;
        if(bitPtr == 8) {
           b = *(++bytePtr);
           bitPtr = 0;
        }
    }

    stream->bitPtr = bitPtr;
    stream->bytePtr = bytePtr;

    return ret;
}
void SkipBits(BitStream* stream, u32 bitCount) {

    auto bitPtr = stream->bitPtr;
    auto bytePtr = stream->bytePtr;

    u32 byteSkip = (bitPtr + bitCount) >> 3;
    bitPtr = (bitPtr + bitCount) & 7;
    stream->bitPtr = bitPtr;
    stream->bytePtr += byteSkip;
}

void FlushByte(BitStream* stream) {

    ASSERT(stream->bitPtr < 8);
    stream->bytePtr += (stream->bitPtr != 0);
    stream->bitPtr = 0;
}


u32 ReverseBits(u32 c, u32 len) {

    u32 reversedCode = 0;
    for(u32 j = 0; j <= len / 2; j++) {

        auto topBitIndex = len - (j + 1);
        u32 bottomBit = (c >> j) & 1;
        u32 topBit = (c >> topBitIndex) & 1;

        reversedCode |= (bottomBit << topBitIndex) | (topBit << j);
    }
    return reversedCode;
}

struct HuffmanEntry {
    u16 symbol;
    u16 bitLen;
};
struct HuffmanDictionary {

    u32 maxCodeLen;
    u32 count;
    HuffmanEntry* entries;
};
HuffmanDictionary AllocateHuffmanDict(LinearAllocator* alloc, u32 maxCodeLen) {

    HuffmanDictionary ret;
    ret.count = 1 << maxCodeLen;
    ret.entries = (HuffmanEntry*)linear_allocate(alloc, ret.count * sizeof(HuffmanEntry));
    ret.maxCodeLen = maxCodeLen;

    return ret;
}
void ComputeHuffmanDict(HuffmanDictionary* dict, u32 count, u32* codeLens) {

    memset(dict->entries, 0, sizeof(HuffmanEntry) * dict->count);
    u32 lenFreqs[16]{};
    for(u32 i = 0; i < count; i++) {
        lenFreqs[codeLens[i]]++;
    }

    u32 code = 0;
    u32 nextCode[16]{};
    lenFreqs[0] = 0;
    for(u32 i = 1; i < 16; i++) {
        code = (code + lenFreqs[i - 1]) << 1;
        nextCode[i] = code;
    }

    for(u32 i = 0; i < count; i++) {

        auto len = codeLens[i];
        if(len) {

            u32 c = nextCode[len];
            nextCode[len]++;
            c = ReverseBits(c, len);
            u32 upperBitCount = dict->maxCodeLen - len;
            for(u32 k = 0; k < (1 << upperBitCount); k++) {
                
                u32 index = (k << len) | c;
                dict->entries[index].bitLen = len;
                dict->entries[index].symbol = i;
            }
        }
    }
}

u32 HuffmanDecode(HuffmanDictionary* dict, BitStream* stream) {

    u32 index = PeakBits(stream, dict->maxCodeLen);
    ASSERT(index < dict->count);
    auto entry = dict->entries + index;
    SkipBits(stream, dict->entries[index].bitLen);
    return dict->entries[index].symbol;
}

void ConstructStaticHuffman(HuffmanDictionary* litLenDict, HuffmanDictionary* distDict) {

    u32 codeLength[288];
    u32 i = 0;
    for(;i < 144; i++) {
        codeLength[i] = 8;
    }
    for(;i < 256; i++) {
        codeLength[i] = 9;
    }
    for(;i < 280; i++) {
        codeLength[i] = 7;
    }
    for(;i < 288; i++) {
        codeLength[i] = 8;
    }
    ComputeHuffmanDict(litLenDict, 288, codeLength);
    for(i = 0;i < 30; i++) {
        codeLength[i] = 5;
    }
    ComputeHuffmanDict(distDict, 30, codeLength);
}
void DynamicHuffman(HuffmanDictionary* dict, HuffmanDictionary* litLen, HuffmanDictionary* dist, BitStream* stream) {

    u32 HLIT = ReadBits(stream, 5) + 257;
    u32 HDIST = ReadBits(stream, 5) + 1;
    u32 HCLEN = ReadBits(stream, 4) + 4;

    constexpr auto maxTableSize = (1 << 5) * 2 + (258);
    const u32 HCLENSwizzle[] = {
        16, 17, 18,0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    };
    u32 litLenDistTable[maxTableSize];
    u32 HCLENTable[19]{};
    ASSERT(HCLEN <= SIZE_OF_ARRAY(HCLENSwizzle));
    
    for(u32 i = 0; i < HCLEN; i++) {
        HCLENTable[HCLENSwizzle[i]] = ReadBits(stream, 3);
    }
    ComputeHuffmanDict(dict, SIZE_OF_ARRAY(HCLENTable), HCLENTable);
    
    u32 lenCount = HLIT + HDIST;
    u32 i = 0;
    for(;i < lenCount;) {

        u32 encodedLen = HuffmanDecode(dict, stream);
        u32 repCount = 1;
        u32 repVal = 0;
        if(encodedLen <= 15) {
            repVal = encodedLen;
        }
        else if(encodedLen == 16) {
            repCount = 3 + ReadBits(stream, 2);
            repVal = litLenDistTable[i - 1];
        }
        else if(encodedLen == 17) {
            repCount = 3 + ReadBits(stream, 3);
        }
        else if(encodedLen == 18) {
            repCount = 11 + ReadBits(stream, 7);
        }
        while(repCount--) {
            litLenDistTable[i++] = repVal;
        }
    }
    ASSERT(i == lenCount);
    ComputeHuffmanDict(litLen, HLIT, litLenDistTable);
    ComputeHuffmanDict(dist, HDIST, litLenDistTable + HLIT);
}

HuffmanEntry lenExtraBits[] = {
    {  3, 0},
    {  4, 0},
    {  5, 0},
    {  6, 0},
    {  7, 0},
    {  8, 0},
    {  9, 0},
    { 10, 0},

    { 11, 1},
    { 13, 1},
    { 15, 1},
    { 17, 1},
    
    { 19, 2},
    { 23, 2},
    { 27, 2},
    { 31, 2},

    { 35, 3},
    { 43, 3},
    { 51, 3},
    { 59, 3},

    { 67, 4},
    { 83, 4},
    { 99, 4},
    {115, 4},

    {131, 5},
    {163, 5},
    {195, 5},
    {227, 5},

    {258, 0},
};
HuffmanEntry distExtraBits[] = {
    {1, 0},
    {2, 0},
    {3, 0},
    {4, 0},
    {5, 1},
    {7, 1},
    {9, 2},
    {13, 2},
    {17, 3},
    {25, 3},
    {33, 4},
    {49, 4},
    {65, 5},
    {97, 5},
    {129, 6},
    {193, 6},
    {257, 7},
    {385, 7},
    {513, 8},
    {769, 8},
    {1025, 9},
    {1537, 9},
    {2049, 10},
    {3073, 10},
    {4097, 11},
    {6145, 11},
    {8193, 12},
    {12289, 12},
    {16385, 13},
    {24577, 13},
};


void LZ77(BitStream* stream, HuffmanDictionary* litLenDict, HuffmanDictionary* distDict, LinearAllocator* out) {


    for(u32 k = 0;; k++) {

        u32 litLen = HuffmanDecode(litLenDict, stream);
        if(litLen < 256) {
            Mem<u8>(linear_allocate(out, 1)) = (u8)litLen;
        }
        else if(litLen > 256) {

            litLen -= 257;
            litLen = lenExtraBits[litLen].symbol + ReadBits(stream, lenExtraBits[litLen].bitLen);
            u32 dist = HuffmanDecode(distDict, stream);
            dist = distExtraBits[dist].symbol + ReadBits(stream, distExtraBits[dist].bitLen);

            auto it = out->base + (out->top - dist);
            for(u32 i = 0; i < litLen; i++) {
                out->base[out->top++] = *it++;
            }
        }
        else {
            break;
        }
    }
}
u32 Inflate(byte* in, LinearAllocator* alloc) {

    BitStream stream{in, 0};
    const u32 top = alloc->top;

    auto CMF = in[0];
    auto FLG = in[1];
    u16 check = (CMF * 256 + FLG);
    ASSERT(check % 31 == 0);

    u32 CM = ReadBits(&stream, 4);
    u32 CINFO = ReadBits(&stream, 4);
    u32 FCHECK = ReadBits(&stream, 5);
    u32 FDICT = ReadBits(&stream, 1);
    u32 FLEVEL = ReadBits(&stream, 2);

    constexpr auto memSize = ((1 << 15) * 4 + (1 << 8)) * sizeof(HuffmanEntry);
    byte mem[memSize];
    auto localAlloc = make_linear_allocator(mem, memSize);

    HuffmanDictionary dict = AllocateHuffmanDict(&localAlloc, 8);
    HuffmanDictionary litLenHuffman = AllocateHuffmanDict(&localAlloc, 15);
    HuffmanDictionary distHuffman = AllocateHuffmanDict(&localAlloc, 15);

    HuffmanDictionary staticLitLenHuffman = AllocateHuffmanDict(&localAlloc, 15);
    HuffmanDictionary staticDistHuffman = AllocateHuffmanDict(&localAlloc, 15);
    ConstructStaticHuffman(&staticLitLenHuffman, &staticDistHuffman);

    bool final;
    do {

        final = ReadBits(&stream, 1);
        u32 type = ReadBits(&stream, 2);
        ASSERT(type != 3);

        switch(type) {
        case 0:
            {
                ASSERT(0);
                FlushByte(&stream);
                u16 len = Mem<u16>(stream.bytePtr);
                u16 nlen = Mem<u16>(stream.bytePtr + 2);
                ASSERT(len == ~nlen);
                stream.bytePtr += 4;

                auto dst = (byte*)linear_allocate(alloc, len);
                for(u32 i = 0; i < len; i++) {
                    dst[i] = *stream.bytePtr++;
                }
                break;
            }
        case 1:
            ASSERT(0);
            LZ77(&stream, &staticLitLenHuffman, &staticDistHuffman, alloc);
            break;
        case 2:
            DynamicHuffman(&dict, &litLenHuffman, &distHuffman, &stream);
            LZ77(&stream, &litLenHuffman, &distHuffman, alloc);
            break;
        }

    } while(!final);

    return alloc->top - top;
}

PNGInfo ParsePNGMemory(byte* pngMemory, LinearAllocator* alloc) {

    ASSERT(pngMemory && alloc);

    byte* mem = pngMemory;
    LOG_ASSERT(VerifyPngSignature(Mem<u64>(mem)), "not png signature");
    mem += 8;

    bool ihdrFound = false;
    u32 chunkCounter = 0;
    PNGInfo info{};

    u64 crc_table[256];
    make_crc_table(crc_table);

    LocalList<PNGComment>* comments = nullptr;
    LocalList<PNGChunk*>* dataChunks = nullptr;

    for(bool run = true; run ;) {

        const auto chunk = (PNGChunk*)mem;
        LOG_ASSERT((chunkCounter == 0) == (chunk->type == PNG_TYPE_STR('I', 'H', 'D', 'R')), "IHDR is not the first chunk");

        switch(chunk->type) {
        case PNG_TYPE_STR('I', 'H', 'D', 'R'):
            {
                
                LOG_ASSERT(!ihdrFound, "multiple IHDR chunks");
                LOG_ASSERT(chunkCounter == 0, "IHDR is not the first chunk");
                LOG_ASSERT(ReverseByteOrder(chunk->length) == sizeof(IHDRPayload), "IHDR payload not 13 bytes");
                ihdrFound = true;

                auto ihdr = Mem<IHDRPayload>(chunk->payload);
                info.height = ReverseByteOrder(ihdr.height);
                info.width = ReverseByteOrder(ihdr.width);
                info.bitDepth = ihdr.bitDepth;
                info.colorType = ihdr.colorType;
                info.filterMethod = ihdr.filterMethod;
                ihdr.interlaceMethod = ihdr.interlaceMethod;

                LOG_ASSERT(PNGCheckColorTypeBitDepth(&info), "invalid bitdepth");
                LOG_ASSERT(ihdr.compressionMethod == 0, "invalid compression method");
                LOG_ASSERT(ihdr.interlaceMethod == 0, "invalid interlace method");
                

                break;
            }
        case PNG_TYPE_STR('I', 'D', 'A', 'T'):
            {
                LocalList<PNGChunk*>* tmp;
                ALLOCATE_LOCAL_LIST(tmp, sizeof(LocalList<PNGChunk*>));
                tmp->item = chunk;
                tmp->next = dataChunks;
                dataChunks = tmp;
                info.dataChunkCount++;

                break;
            }
        case PNG_TYPE_STR('P', 'L', 'T', 'E'):
            {
                LOG_ASSERT(ReverseByteOrder(chunk->length) % 3 == 0, "PLTE chunk length not divisible by 3");
                info.paletteColorCount = chunk->length / 3;
                info.paletteColors = (u16*)chunk->payload;
                break;
            }
        case PNG_TYPE_STR('g', 'A', 'M', 'A'):
            {
                info.gamma = ReverseByteOrder(Mem<u32>(chunk->payload));
                break;
            }
        case PNG_TYPE_STR('c', 'H', 'R', 'M'):
            {
                info.chroma = LoadChroma(chunk->payload);
                break;
            }
        case PNG_TYPE_STR('b', 'K', 'G', 'D'):
            {
                if(info.colorType == 3) {
                    info.bkgColor = chunk->payload[0];
                }
                else if(info.colorType == 0 || info.colorType == 4) {
                    info.bkgColor = ReverseByteOrder(Mem<u16>(chunk->payload));
                }
                else if(info.colorType == 2 || info.colorType == 6) {
                    u16 red     = ReverseByteOrder(Mem<u16>(chunk->payload + 0));
                    u16 green   = ReverseByteOrder(Mem<u16>(chunk->payload + 2));
                    u16 blue    = ReverseByteOrder(Mem<u16>(chunk->payload + 4));
                    info.bkgColor = (u64)red << 32 | (u64)green << 16 | (u64)blue;
                }
                break;
            }
        case PNG_TYPE_STR('I', 'E', 'N', 'D'):
            {
                run = false;
                break;
            }
        case PNG_TYPE_STR('t', 'E', 'X', 't'):
            {

                auto keyWordLen = str_len((const char*)chunk->payload);
                auto chunkS = ReverseByteOrder(chunk->length);
                auto textLen = chunkS - keyWordLen;

                LocalList<PNGComment>* tmp;
                ALLOCATE_LOCAL_LIST(tmp, sizeof(LocalList<PNGComment>));
                tmp->item.keyWord = (const char*)chunk->payload;
                tmp->item.keyWordSize = keyWordLen;
                tmp->item.textSize = textLen;
                tmp->next = comments;
                comments = tmp;
                info.commentCount++;
                break;
            }
        default:
            {
                break;
            }
        }

        auto computedCRC = compute_crc(crc_table, (byte*)&chunk->type, ReverseByteOrder(chunk->length) + 4);
        u32* end = (u32*)(chunk->payload + ReverseByteOrder(chunk->length));
        u32 CRC = ReverseByteOrder(*end);
        LOG_ASSERT(CRC == computedCRC, "CRC corruption");

        chunkCounter++;
        mem = (byte*)(end + 1);
    }

    info.comments = (PNGComment*)linear_allocate(alloc, info.commentCount * sizeof(PNGComment));
    u32 i = 0;
    for(auto it = comments; it; it = it->next) {
        info.comments[i++] = it->item;
    }

    info.dataChunks = (PNGChunk**)linear_allocate(alloc, info.dataChunkCount * sizeof(PNGChunk*));
    i = 1;
    for(auto it = dataChunks; it; it = it->next) {
        info.dataChunks[info.dataChunkCount - i] = it->item;
        i++;
    }

    if(info.dataChunkCount) {

        auto CMF = info.dataChunks[0]->payload[0];
        auto FLG = info.dataChunks[0]->payload[1];
        u16 check = (CMF * 256 + FLG);
        ASSERT(check % 31 == 0);

        u8 CM = CMF & 0xF;
        u8 CINFO = CMF >> 4;
        u8 FCHECK = FLG & 0x1F;
        u8 FDICT = (FLG >> 5) & 1;
        u8 FLEVEL = (FLG >> 6);
        ASSERT(CM == 8);
    }

    return info;
}

u32 PNGDataSize(PNGInfo* png) {

    u32 size = 0;
    for(u32 i = 0; i < png->dataChunkCount; i++) {
        auto chunk = png->dataChunks[i];
        size += ReverseByteOrder(chunk->length);
    }
    return size;
}
constexpr u32 PixelChannelCounts[] = {
    1,
    0,
    3,
    1,
    2,
    0,
    4,
};
u8 PaethPredictor(u32 a, u32 b, u32 c) {

    u32 p = a + b - c;
    u32 pa = Abs(p - a);
    u32 pb = Abs(p - b);
    u32 pc = Abs(p - c);

    u8 Pr;
    if (pa <= pb && pa <= pc) {
        Pr = a;
    }
    else if (pb <= pc) {
        Pr = b;
    }
    else {
        Pr = c;
    }
    return Pr;
}
u32 PNGReconstructFilter(PNGInfo* info, u8* dst, u8* src, u32 outChannelN) {

    ASSERT(info->filterMethod == 0);

    u32 channelSize = info->bitDepth / 8;
    u32 pixelSize = channelSize * PixelChannelCounts[info->colorType];
    u32 scanLineSize = info->width * pixelSize;

    i32 filterByteCount = pixelSize;
    i32 outputByteCount = outChannelN * channelSize;

    u8 null[outputByteCount+1]{};
    u8* priorRow = null + outputByteCount;

    const auto dstBegin = dst;
    const auto srcBegin = src;

    for(u32 i = 0; i < info->height; i++) {

        
        u8 filterType = *src++;
        ASSERT(filterType < 5);

        u8* currentRow = dst;
        for (u32 k = 0; k < pixelSize; ++k) {
            switch(filterType) {
                case 0: *dst = *src; break;
                case 1: *dst = *src; break;
                case 2: *dst = (*src + *priorRow) & 255; break;
                case 3: *dst = (*src + *priorRow >> 1) & 255; break;
                case 4: *dst = (*src + PaethPredictor(0,*priorRow,0)) & 255; break;
            }
            dst++;
            src++;
            priorRow += (i != 0);
        }

        if(outChannelN == 4) {
            *dst++ = 255;
            priorRow += (i != 0);
        }

        #define BYTECAST(x)  ((u8) ((x) & 255))  // truncate int to byte without warnings
        for (i32 j = info->width - 1; j >= 1;) {

            for (i32 k = 0; k < filterByteCount; ++k) {

                switch (filterType) {
                case 0:
                    dst[k] = src[k];
                    break;
                case 1:
                    dst[k] = BYTECAST(src[k] + dst[k-outputByteCount]);
                    break;
                case 2:
                    dst[k] = BYTECAST(src[k] + priorRow[k]);
                    break;
                case 3:
                    dst[k] = BYTECAST(src[k] + ( (priorRow[k] + dst[k-outputByteCount]) >> 1 ));
                    break;
                case 4:
                    dst[k] = BYTECAST(src[k] + PaethPredictor(dst[k-outputByteCount], priorRow[k], priorRow[k-outputByteCount]));
                    break;
                }
            }
            --j;
            dst[filterByteCount] = 255;
            src += filterByteCount;
            dst += outputByteCount;
            priorRow += (i == 0 ? 0 : outputByteCount);
        }

        /*
        for(i32 j = 0; j < (info->width * pixelSize) - pixelSize; j++) {

            switch (filterType) {
            case 0:
                *dst = *src;
                break;
            case 1:
                *dst = BYTECAST(*src + dst[-outputByteCount]);
                break;
            case 2:
                *dst = BYTECAST(*src + *priorRow);
                break;
            case 3:
                *dst = BYTECAST(*src + ( (*priorRow + dst[-outputByteCount]) >> 1 ));
                break;
            case 4:
                *dst = BYTECAST(*src + PaethPredictor(dst[-outputByteCount], *priorRow, priorRow[-outputByteCount]));
                break;
            }

            src++;
            dst++;
            priorRow += (i != 0);

            if(outChannelN == 4 && (dst - dstBegin) % 4 == 0) {
                auto off = dst - dstBegin;
                dst++;
            }
        }
        */


        priorRow = currentRow;

    }
    
    return dst - dstBegin;
}
enum {
   STBI__F_none=0,
   STBI__F_sub=1,
   STBI__F_up=2,
   STBI__F_avg=3,
   STBI__F_paeth=4,
   // synthetic filters used for first scanline to avoid needing a dummy row of 0s
   STBI__F_avg_first,
   STBI__F_paeth_first
};

u8 first_row_filter[5] = {
   STBI__F_none,
   STBI__F_sub,
   STBI__F_none,
   STBI__F_avg_first,
   STBI__F_paeth_first
};

i32 PaethPredictor(i32 a, i32 b, i32 c) {
   
   i32 p = a + b - c;
   i32 pa = abs(p-a);
   i32 pb = abs(p-b);
   i32 pc = abs(p-c);
   if (pa <= pb && pa <= pc) return a;
   if (pb <= pc) return b;
   return c;
}

u32 PNGReconstructFilter_(PNGInfo* info, u8* dst, u8* src, u32 outN) {
   
    i32 bytes = (info->bitDepth == 16? 2 : 1);
    i32 stride = info->width * outN * bytes;

    i32 imgN = PixelChannelCounts[info->colorType];
    i32 img_width_bytes = (((imgN * info->width * info->bitDepth) + 7) >> 3);
    i32 filter_bytes = imgN * bytes;
    i32 width = info->width;
    i32 output_bytes = outN * bytes;

    u8* const dstBegin = dst;
    u8* const srcBegin = src;

    ASSERT( outN == imgN || outN == imgN + 1);
    i32 imgLen = (img_width_bytes + 1) * info->height;

    for (i32 j = 0; j < info->height; ++j) {
      
        u8* cur = dstBegin + stride * j;
        u8* prior;

        i32 filter = *src++;
        ASSERT(filter < 5);
       
        if (info->bitDepth < 8) {
            ASSERT(img_width_bytes <= info->width);
            cur += info->width * outN - img_width_bytes; // store output to the rightmost img_len bytes, so we can decode in place
            filter_bytes = 1;
            width = img_width_bytes;
        }

        prior = cur - stride; // bugfix: need to compute this after 'cur +=' computation above

        // if first row, use special filter that doesn't sample previous row
        if (j == 0) filter = first_row_filter[filter];

        // handle first byte explicitly
        for (i32 k = 0; k < filter_bytes; ++k) {
            switch (filter) {
                case STBI__F_none       : cur[k] = src[k]; break;
                case STBI__F_sub        : cur[k] = src[k]; break;
                case STBI__F_up         : cur[k] = BYTECAST(src[k] + prior[k]); break;
                case STBI__F_avg        : cur[k] = BYTECAST(src[k] + (prior[k]>>1)); break;
                case STBI__F_paeth      : cur[k] = BYTECAST(src[k] + PaethPredictor(0,prior[k],0)); break;
                case STBI__F_avg_first  : cur[k] = src[k]; break;
                case STBI__F_paeth_first: cur[k] = src[k]; break;
            }
        }

        if(info->bitDepth == 8) {
            if (imgN != outN) {
                cur[imgN] = 255; // first pixel
            }
            src += imgN;
            cur += outN;
            prior += outN;
        }
        else if (info->bitDepth == 16) {
            if (imgN != outN) {
                cur[filter_bytes]   = 255; // first pixel top byte
                cur[filter_bytes+1] = 255; // first pixel bottom byte
            }
            src += filter_bytes;
            cur += output_bytes;
            prior += output_bytes;
        }
        else {
            src += 1;
            cur += 1;
            prior += 1;
        }

        // this is a little gross, so that we don't switch per-pixel or per-component
        if (info->bitDepth < 8 || imgN == outN) {
            
            int nk = (width - 1) * filter_bytes;

            #define STBI__CASE(f) \
                case f:     \
                    for (i32 k = 0; k < nk; ++k)
            switch (filter) {
                // "none" filter turns into a memcpy here; make that explicit.
                case STBI__F_none:
                    memcpy(cur, src, nk);
                    break;
                case STBI__F_sub:
                    for (i32 k = 0; k < nk; ++k) {
                        cur[k] = BYTECAST(src[k] + cur[k - filter_bytes]);
                    }
                    break;
                case STBI__F_up:
                    for (i32 k = 0; k < nk; ++k) {
                        cur[k] = BYTECAST(src[k] + prior[k]);
                    }
                    break;
                case STBI__F_avg:
                    for (i32 k = 0; k < nk; ++k) {
                        cur[k] = BYTECAST(src[k] + ((prior[k] + cur[k - filter_bytes])>>1));
                    }
                    break;
                case STBI__F_paeth:
                    for (i32 k = 0; k < nk; ++k) {
                        cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k - filter_bytes],prior[k],prior[k - filter_bytes]));
                    }
                    break;
                case STBI__F_avg_first:
                    for (i32 k = 0; k < nk; ++k) {
                        cur[k] = BYTECAST(src[k] + (cur[k - filter_bytes] >> 1));
                    }
                    break;
                case STBI__F_paeth_first:
                    for (i32 k = 0; k < nk; ++k) {
                        cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k - filter_bytes],0,0));
                    }
                    break;
            }
            #undef STBI__CASE
            src += nk;
        }  else {
        
            ASSERT(imgN + 1 == outN);
            #define STBI__CASE(f) \
                case f:     \
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) \
                    for (i32 k = 0; k < filter_bytes; ++k)
            switch (filter) {
                STBI__CASE(STBI__F_none)         { cur[k] = src[k]; } break;
                case STBI__F_sub:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                            cur[k] = BYTECAST(src[k] + cur[k - output_bytes]);
                        }
                    }
                    break;
                //STBI__CASE(STBI__F_sub)          { cur[k] = BYTECAST(src[k] + cur[k - output_bytes]); } break;
                STBI__CASE(STBI__F_up)           { cur[k] = BYTECAST(src[k] + prior[k]); } break;
                STBI__CASE(STBI__F_avg)          { cur[k] = BYTECAST(src[k] + ((prior[k] + cur[k- output_bytes])>>1)); } break;
                STBI__CASE(STBI__F_paeth)        { cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k- output_bytes],prior[k],prior[k- output_bytes])); } break;
                STBI__CASE(STBI__F_avg_first)    { cur[k] = BYTECAST(src[k] + (cur[k- output_bytes] >> 1)); } break;
                STBI__CASE(STBI__F_paeth_first)  { cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k- output_bytes],0,0)); } break;
            }
            #undef STBI__CASE

            // the loop above sets the high byte of the pixels' alpha, but for
            // 16 bit png files we also need the low byte set. we'll do that here.
            if(info->bitDepth == 16) {
                cur = dstBegin + stride*j; // start at the beginning of the row again
                for (i32 i = 0; i < info->width; ++i,cur += output_bytes) {
                cur[filter_bytes + 1] = 255;
                }
            }
        }
    }

    return info->width * info->height * outN;
}

u32 MemCmp(byte* p0, byte* p1 , u32 size) {

    for(u32 i = 0; i < size; i++) {
        if(p0[i] != p1[i]) {
            return i;
        }
    }
    return size;
}
ImageDescriptor MakeImagePNG(byte* pngMemory, LinearAllocator* alloc) {

    const auto begin = alloc->top;

    auto info = ParsePNGMemory(pngMemory, alloc);
    u64 compressedImgSize = PNGDataSize(&info);
    auto compressedImg = (byte*)linear_allocate(alloc, compressedImgSize);
    auto dst = compressedImg;
    
    for(u32 i = 0; i < info.dataChunkCount; i++) {
        u32 payloadSize = ReverseByteOrder(info.dataChunks[i]->length);
        memcpy(dst, info.dataChunks[i]->payload, payloadSize);
        dst += payloadSize;
    }

    auto decompressed = (byte*)linear_allocator_top(alloc);
    auto decompressedSize = Inflate(compressedImg, alloc);

    u8* pixelData = (u8*)linear_allocator_top(alloc);
    auto imgSize = PNGReconstructFilter(&info, pixelData, decompressed, 4);
    linear_allocate(alloc, imgSize);

    u8* stbi = (u8*)linear_allocator_top(alloc);
    auto stbiSize = PNGReconstructFilter_(&info, stbi, decompressed, 4);
    ASSERT(stbiSize == imgSize);

    auto cmp = MemCmp(stbi, pixelData, imgSize);

    return {};
    //alloc->top = begin;
    //memcpy(linear_allocator_top(alloc), pixelData, imgSize);
    
    /*
    ImageDescriptor descriptor;    
    descriptor.height = info.height;
    descriptor.width = info.width;
    descriptor.img = (Pixel*)linear_allocate(alloc, imgSize);
    */

    //return descriptor;
}
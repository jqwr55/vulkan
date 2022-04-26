#include <common.h>
#include <cstddef>
#include <memory>
#include <math.h>


byte* global_malloc_base;
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
u32 AddBinary(bool* a, u32 aCount, bool* b, u32 bCount, bool* result) {

    i32 s = 0;
    i32 i = aCount - 1;
    i32 j = bCount - 1;
    u32 ret = 0;
    while (i >= 0 || j >= 0 || s == 1) {      
        // Compute sum of last digits and carry
        s += (i >= 0 ? a[i] : 0);
        s += (j >= 0 ? b[j] : 0);
        // If current digit sum is 1 or 3, add 1 to result
        result[ret++] = s & 1;
        // Compute carry
        s /= 2;
        // Move to next digits
        i--;
        j--;
    }

    // flip
    for(u32 i = 0; i < ret/2; i++) {
        result[i] = result[ret - i - 1];
    }

    return ret;
}
u32 IncBinary(bool* a, u32 aCount, bool* result) {

    i32 s = 0;
    i32 i = aCount - 1;
    i32 j = 0;
    u32 ret = 0;
    while (i >= 0 || j >= 0 || s == 1) {      
        // Compute sum of last digits and carry
        s += (i >= 0 ? a[i] : 0);
        s += (j >= 0 ? 1 : 0);
        // If current digit sum is 1 or 3, add 1 to result
        result[ret++] = s & 1;
        // Compute carry
        s /= 2;
        // Move to next digits
        i--;
        j--;
    }

    // flip
    for(u32 i = 0; i < ret/2; i++) {
        result[i] = result[ret - i - 1];
    }

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
    buffer_size -= 3;

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
void local_printer(Printer* printer, const char *format, ...) {

    va_list args;
    va_start(args, format);
    auto end = print_fn_v(printer->io->base + printer->io->top, printer->io->cap - printer->io->top, format, args);
    va_end(args);
    
    if( (end - printer->io->base) >= printer->io->cap) {
        
        printer->flush(printer->user, printer->io);
        auto top = (byte*)linear_allocator_top(printer->io);
        va_start(args, format);
        end = print_fn_v(top, linear_allocator_free_size(printer->io), format, args);
        va_end(args);

        ASSERT(end != printer->io->base + printer->io->cap);
    }
}


u64 ReadFile(const char* fileName, byte* buffer, u32 bufferSize) {

    FILE* file = fopen(fileName, "r");
    u64 size = ~u64(0);
    if(file) {

        fseek(file, 0, SEEK_END);
        size = ftell(file);
        fseek(file, 0, SEEK_SET);
        if(size > bufferSize) {
            fclose(file);
            return ~u64(0);
        }

        fread(buffer, size, 1, file);
        fclose(file);
    }

    return size;
}
u64 ReadFile(const char* fileName, LinearAllocator* mem) {

    FILE* file = fopen(fileName, "r");
    u64 size = ~u64(0);
    if(file) {

        fseek(file, 0, SEEK_END);
        size = ftell(file);
        fseek(file, 0, SEEK_SET);

        auto buffer = (byte*)linear_allocate(mem, size);

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
        local_printer(&printer, "s", "signature missmatch\n");
    }

    bool firstIdat = true;
    const byte* mem = pngMemory + 8;
    for(bool run = true; run ;) {

        auto chunk = (PNGChunk*)mem;
        switch(chunk->type) {
        case PNG_TYPE_STR('I', 'H', 'D', 'R'):
            {
                
                local_printer(&printer, "xs", mem - pngMemory, " IHDR\n");

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

                local_printer(&printer, "suc", " height: ", (u64)ihdr.height, '\n');
                local_printer(&printer, "suc", " width: ", (u64)ihdr.width, '\n');
                local_printer(&printer, "suc", " bitDepth: ", (u64)ihdr.bitDepth, '\n');
                local_printer(&printer, "sucss", " colorType: ", (u64)ihdr.colorType, '(', colorTypeStrs[ihdr.colorType], ")\n");
                local_printer(&printer, "su", " compressionMethod: ", (u64)ihdr.compressionMethod);
                if(ihdr.compressionMethod == 0) {
                    local_printer(&printer, "s", "(deflate)\n");
                }
                local_printer(&printer, "suc", " filterMethod: ", (u64)ihdr.filterMethod, '\n');
                local_printer(&printer, "suc", " interlaceMethod: ", (u64)ihdr.interlaceMethod, '\n');
                break;
            }
        case PNG_TYPE_STR('I', 'D', 'A', 'T'):
            {
                local_printer(&printer, "xs", mem - pngMemory, " IDAT\n");

                if(firstIdat) {
                    firstIdat = false;

                    auto CMF = chunk->payload[0];
                    auto FLG = chunk->payload[1];

                    u16 check = (CMF * 256 + FLG);
                    if(check % 31 != 0) {
                        local_printer(&printer, "s", " zlib corruption: CMF*256 + FLG is not multiple of 31\n");
                    }

                    u8 CM = CMF & 0xF;
                    u8 CINFO = CMF >> 4;
                    u8 FCHECK = FLG & 0x1F;
                    u8 FDICT = (FLG >> 5) & 1;
                    u8 FLEVEL = (FLG >> 6);

                    local_printer(&printer, "suc", " CM: ", (u64)CM, '\n');
                    local_printer(&printer, "suc", " CINFO: ", (u64)CINFO, '\n');
                    local_printer(&printer, "suc", " FCHECK: ", (u64)FCHECK, '\n');
                    local_printer(&printer, "suc", " FDICT: ", (u64)FDICT, '\n');
                    local_printer(&printer, "suc", " FLEVEL: ", (u64)FLEVEL, '\n');
                }

                break;
            }
        case PNG_TYPE_STR('I', 'E', 'N', 'D'):
            {
                local_printer(&printer, "xs", mem - pngMemory, " IEND\n");
                run = false;
                break;
            }
        case PNG_TYPE_STR('t', 'E', 'X', 't'):
            {

                auto keyWordLen = str_len((const char*)chunk->payload);
                auto chunkS = ReverseByteOrder(chunk->length);
                auto textLen = chunkS - keyWordLen;
                local_printer(&printer, "xss*cs*c", mem - pngMemory, " tEXt\n ", chunk->payload, keyWordLen, ' ', chunk->payload + keyWordLen, textLen, '\n');

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
                local_printer(&printer, "xcs*c", mem - pngMemory, ' ', &chunk->type, 4, '\n');
                break;
            }
        default:
            {
                local_printer(&printer, "xss*c", mem - pngMemory, " unkonw chunk: ", &chunk->type, 4, '\n');
                break;
            }
        }

        u32* end = (u32*)(chunk->payload + ReverseByteOrder(chunk->length));
        mem = (byte*)(end + 1);
    }

    printer.flush(printer.user, printer.io);
}


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
u32 PeakBitsReversed(const BitStream* stream, u32 bitCount) {

    ASSERT(bitCount < 33);

    auto bitPtr = stream->bitPtr;
    auto bytePtr = stream->bytePtr;

    bytePtr += bitPtr >> 3;
    bitPtr &= 7;

    byte b = *bytePtr;
    if(b == 0xFF) {
        bytePtr++;
        byte fill = *(bytePtr++);
        while(fill == 0xFF) {
            fill = *(bytePtr++);
        }
        ASSERT(fill == 0);
    }
    u32 ret = 0;
    for(u32 i = 0; i < bitCount; i++) {
        
        u32 bit = (b >> (7 - bitPtr)) & 1;
        ret |= bit << (31-i);

        bitPtr++;
        if(bitPtr == 8) {
            b = *(++bytePtr);

            if(b == 0xFF) {
                bytePtr++;
                byte fill = *(bytePtr++);
                while(fill == 0xFF) {
                    fill = *(bytePtr++);
                }
                ASSERT(fill == 0);
            }

            bitPtr = 0;
        }
    }

    return ret;
}

#define stbi_lrot(x,y) (((x) << (y)) | ((x) >> (-(y) & 31)))

void GrowBitBuffJPEG(BitStream* stream) {

    do {
        u32 b = *stream->bytePtr++;
        if (b == 0xFF) {
            u32 c = *stream->bytePtr++;
            while (c == 0xFF) {
                c = *stream->bytePtr++;
            }
            
        }
        stream->bitBuff |= b << (24 - stream->bitCnt);
        stream->bitCnt += 8;
    } while (stream->bitCnt <= 24);
}
u32 ReadBitsJPEG(BitStream* stream, u32 bitCnt) {

    ASSERT(bitCnt < 33);
    if(stream->bitCnt < bitCnt) {
        GrowBitBuffJPEG(stream);
    }

    u32 mask = (1 << bitCnt) - 1;

    u32 k = stbi_lrot(stream->bitBuff, bitCnt);
    stream->bitBuff = k & (~mask);
    k &= mask;
    stream->bitCnt -= bitCnt;
    return k;
}
u32 PeakBitsJPEG(const BitStream* stream, u32 bitCnt) {

    ASSERT(bitCnt < 33);
    auto copy = *stream;
    if(copy.bitCnt < bitCnt) {
        GrowBitBuffJPEG(&copy);
    }

    u32 mask = (1 << bitCnt) - 1;
    u32 ret = stbi_lrot(copy.bitBuff, bitCnt);
    ret &= mask;
    return ret;
}
void SkipBitsJPEG(BitStream* stream, u32 bitCnt) {


    if(stream->bitCnt < bitCnt) {
        GrowBitBuffJPEG(stream);
    }
    stream->bitBuff <<= bitCnt;
    stream->bitCnt -= bitCnt;
}

/*
u32 ReadBitsReversed(BitStream* stream, u32 bitCount) {

    ASSERT(bitCount < 33);

    auto bitPtr = stream->bitPtr;
    auto bytePtr = stream->bytePtr;

    bytePtr += bitPtr >> 3;
    bitPtr &= 7;

    byte b = *bytePtr;
    if(b == 0xFF) {
        bytePtr++;
        byte fill = *(bytePtr++);
        while(fill == 0xFF) {
            fill = *(bytePtr++);
        }
        ASSERT(fill == 0);
    }
    u32 ret = 0;
    for(u32 i = 0; i < bitCount; i++) {
        
        u32 bit = (b >> (7 - bitPtr)) & 1;
        ret |= bit << (31-i);

        bitPtr++;
        if(bitPtr == 8) {
            b = *(++bytePtr);
            if(b == 0xFF) {
                bytePtr++;
                byte fill = *(bytePtr++);
                while(fill == 0xFF) {
                    fill = *(bytePtr++);
                }
                ASSERT(fill == 0);
            }
            bitPtr = 0;
        }
    }

    stream->bitPtr = bitPtr;
    stream->bytePtr = bytePtr;

    return ret;
}
*/
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

template<typename T>
void PrintBits(T c) {

    constexpr u32 BIT_COUNT = sizeof(T)*8;
    for(u32 i = 0; i < BIT_COUNT; i++) {

        auto bit = (c >> ((BIT_COUNT-1) - i)) & 1;
        global_print("u", (u32)bit);
    }
    global_print("c", '\n');
    global_io_flush();
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
                auto e = dict->entries + index;
                ASSERT(dict->entries[index].bitLen == 0);
                
                dict->entries[index].bitLen = len;
                dict->entries[index].symbol = i;
            }
        }
    }
}
void PrintHuffmanTableJPEG(u8* bits, u8* symbols) {

    u32 sizes[257];
    u32 sizeCount = 0;

    for (u32 i = 0; i < 16; ++i) {
        for (u32 j = 0; j < bits[i]; ++j) {
            sizes[sizeCount++] = i+1;
        }
    }
    sizes[sizeCount] = 0;

    u32 lenFreqs[17]{};
    for(u32 i = 0; i < sizeCount; i++) {
        lenFreqs[sizes[i]]++;
    }
    u32 code = 0;
    u32 nextCode[17]{};
    for(u32 i = 1; i < SIZE_OF_ARRAY(nextCode); i++) {
        code = (code + lenFreqs[i - 1]) << 1;
        nextCode[i] = code;
    }

    global_print("s", "symbol\tcode\tlen\n");
    u32 symbolIndex = 0;
    for(u32 i = 0; i < sizeCount; i++) {
        
        auto len = sizes[i];
        if(len) {

            u32 c = nextCode[len]++;
            u16 symbol = (u16)symbols[symbolIndex++];
            global_print("ucucuc", symbol, '\t', c, '\t', len, '\n');
        }
    }    
    global_print("c", '\n');
    global_io_flush();
}
void ComputeHuffmanTableJPEG(HuffmanDictionary* dict, u8* bits, u8* symbols) {

    memset(dict->entries, 0, sizeof(HuffmanEntry) * dict->count);
    u32 sizes[257];
    u32 sizeCount = 0;

    for (u32 i = 0; i < 16; ++i) {
        for (u32 j = 0; j < bits[i]; ++j) {
            sizes[sizeCount++] = i+1;
        }
    }
    sizes[sizeCount] = 0;

    u32 lenFreqs[17]{};
    for(u32 i = 0; i < sizeCount; i++) {
        lenFreqs[sizes[i]]++;
    }
    u32 code = 0;
    u32 nextCode[17]{};
    for(u32 i = 1; i < SIZE_OF_ARRAY(nextCode); i++) {
        code = (code + lenFreqs[i - 1]) << 1;
        nextCode[i] = code;
    }

    u32 symbolIndex = 0;
    for(u32 i = 0; i < sizeCount; i++) {
        
        auto len = sizes[i];
        if(len) {

            u32 c = nextCode[len]++;
            u16 symbol = (u16)symbols[symbolIndex++];
            u32 lowerBitCount = dict->maxCodeLen - len;

            for(u32 k = 0; k < (1 << lowerBitCount); k++) {
                
                u32 index = (c << lowerBitCount) | k;
                auto e = dict->entries + index;
                ASSERT(dict->entries[index].bitLen == 0);

                dict->entries[index].bitLen = len;
                dict->entries[index].symbol = symbol;
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
u32 HuffmanDecodeJPEG(HuffmanDictionary* dict, BitStream* stream) {

    u32 index = PeakBitsJPEG(stream, dict->maxCodeLen);
    ASSERT(index < dict->count);
    auto entry = dict->entries + index;
    SkipBitsJPEG(stream, dict->entries[index].bitLen);
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
                ALLOCATE_LOCAL_LIST(tmp);
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
                ALLOCATE_LOCAL_LIST(tmp);
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
i32 PaethPredictor(i32 a, i32 b, i32 c) {
   
   i32 p = a + b - c;
   i32 pa = Abs(p-a);
   i32 pb = Abs(p-b);
   i32 pc = Abs(p-c);
   if (pa <= pb && pa <= pc) return a;
   if (pb <= pc) return b;
   return c;
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
                    {
                        auto src_v = src[k];
                        auto predictor = PaethPredictor(dst[k-outputByteCount], priorRow[k], priorRow[k-outputByteCount]);
                        dst[k] = BYTECAST(src_v + predictor );
                    }
                    break;
                }
            }
            --j;
            dst[filterByteCount] = 255;
            src += filterByteCount;
            dst += outputByteCount;
            priorRow += (i == 0 ? 0 : outputByteCount);
        }


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
                case STBI__F_none:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                             cur[k] = src[k];
                        }
                    }
                    break;
                case STBI__F_sub:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                            cur[k] = BYTECAST(src[k] + cur[k - output_bytes]);
                        }
                    }
                    break;
                case STBI__F_up:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                            cur[k] = BYTECAST(src[k] + prior[k]);
                        }
                    }
                    break;
                case STBI__F_avg:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                            cur[k] = BYTECAST(src[k] + ((prior[k] + cur[k- output_bytes])>>1));
                        }
                    }
                    break;
                case STBI__F_paeth:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        const auto offf = cur - dstBegin;
                        for (i32 k = 0; k < filter_bytes; ++k) {

                            cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k- output_bytes],prior[k],prior[k- output_bytes]));
                        }
                    }
                    break;
                case STBI__F_avg_first:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                            cur[k] = BYTECAST(src[k] + (cur[k- output_bytes] >> 1));
                        }
                    }
                    break;
                case STBI__F_paeth_first:
                    for (i32 i = info->width - 1; i >= 1; --i, cur[filter_bytes] = 255, src += filter_bytes, cur += output_bytes, prior += output_bytes) {
                        for (i32 k = 0; k < filter_bytes; ++k) {
                            cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k- output_bytes],0,0));
                        }
                    }
                    break;
                    /*
                STBI__CASE(STBI__F_none)         { cur[k] = src[k]; } break;
                STBI__CASE(STBI__F_sub)          { cur[k] = BYTECAST(src[k] + cur[k - output_bytes]); } break;
                STBI__CASE(STBI__F_up)           { cur[k] = BYTECAST(src[k] + prior[k]); } break;
                STBI__CASE(STBI__F_avg)          { cur[k] = BYTECAST(src[k] + ((prior[k] + cur[k- output_bytes])>>1)); } break;
                STBI__CASE(STBI__F_paeth)        { cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k- output_bytes],prior[k],prior[k- output_bytes])); } break;
                STBI__CASE(STBI__F_avg_first)    { cur[k] = BYTECAST(src[k] + (cur[k- output_bytes] >> 1)); } break;
                STBI__CASE(STBI__F_paeth_first)  { cur[k] = BYTECAST(src[k] + PaethPredictor(cur[k- output_bytes],0,0)); } break;
                    */
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

u32 MemCmp(void* p0, void* p1 , u32 size) {

    byte* s0 = (byte*)p0;
    byte* s1 = (byte*)p1;

    for(u32 i = 0; i < size; i++) {
        if(s0[i] != s1[i]) {
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
    
    alloc->top = begin;
    ImageDescriptor descriptor;
    descriptor.height = info.height;
    descriptor.width = info.width;
    descriptor.img = (Pixel*)linear_allocate(alloc, info.height * info.width * 4);
    for(u32 i = 0; i < info.height * info.width * 4; i++) {
        ((byte*)descriptor.img)[i] = pixelData[i];
    }

    return descriptor;
}



APP0Payload* VerifyJFIFignature(const byte* mem) {

    JPEGArithmeticCondTable t;

    auto SOI = (JPEGMarker*)mem;
    auto APP = (JPEGMarkerSegment*)(mem + 2);

    constexpr u32 signature = u32(0xE0) << 24 | u32(0xFF) << 16 | u32(0xD8) << 8 | u32(0xFF);
    if(Mem<u32>((u32*)mem) != signature) {
        return nullptr;
    }

    auto len = ReverseByteOrder(APP->len);
    ASSERT(len > 0 && len < 255);
    len -= sizeof(APP0Payload);
    if(len % 3 != 0) {
        return nullptr;
    }

    auto appInfo = (APP0Payload*)APP->payload;
    if(!str_cmp("JFIF", appInfo->identifier)) {
        return nullptr;
    }

    if(appInfo->densityUnit > 2) {
        return nullptr;
    }

    if(appInfo->xDensity == 0 || appInfo->yDensity == 0) {
        return nullptr;
    }

    if(len) {
        len == (appInfo->yThumbnail * appInfo->xThumbnail * 3) ? appInfo : nullptr;
    }

    return appInfo;
}

JPEGMarker* GetMarker(const byte* mem) {
   
    auto it = (JPEGMarker*)mem;
    while(it->signature != 0xFF || it->type == 0) {

        mem += (it->type == 0) + 1;
        it = (JPEGMarker*)mem;
    }

    return (JPEGMarker*)mem;
}
byte* PrintJPEGQT(byte* begin, byte* mem) {
    
    auto qt = (JPEGQuantizationTable*)mem;
    u32 len = ReverseByteOrder(qt->len) - 2;
    global_print("xsu", mem - begin, " Qunatization table\n len ", len);
    for(u32 i = 0; i < len; ) {
        auto param = (JPEGQuantizationParameter*)((byte*)qt->params) + i;

        global_print("su sucus",
            "\n  destination identifier ", (u32)param->dstIdentifier,
            "\n  element precision ", (u32)param->elemPrecision, '(', (param->elemPrecision ? 16 : 8),
            ")\n  values[64] "
        );

        u32 de_zig_zaged[64];
        for(u32 k = 0; k < 64; k++) {
            u32 zig = jpeg_de_zig_zag[k];
            u64 q;
            if(param->elemPrecision) {
                q = (u32)Mem<u16>(param->Q + k*2);
            }
            else {
                q = param->Q[k];
            }
            de_zig_zaged[zig] = q;
        }

        for(u32 k = 0; k < 8; k++) {
            
            global_print("s", "\n  ");
            for(u32 j = 0; j < 8; j++) {
            
                auto index = k * 8 + j;
                global_print("uc", de_zig_zaged[index], '\t');
            }
        }
        global_print("c", '\n');

        u32 ad = ((u32)param->elemPrecision + 1) * 64;
        i += ad + 1;
    }
    
    return mem = ((byte*)qt->params) + len;
}
byte* PrintJPEGHuffmanTable(byte* begin, byte* mem) {

    auto huff = (JPEGHuffmanTable*)mem;
    u32 len = ReverseByteOrder(huff->len) - 2;
    global_print("xsu", mem - begin, " Huffman table\n len ", len);

    for(u32 i = 0; i < len;) {

        auto param = (JPEGHuffmanParameter*)(((byte*)huff->params) + i);
        u8 bb = Mem<u8>(param);

        global_print("su su su s",
            "\n  Huffman table destination identifier ", (u32)param->huffDstIdentifier,
            "\n  table class ", (u32)param->tableClass,
            "\n  symbol count ", len - 17,
            "\n  lengths[16] "
        );
        for(u32 k = 0; k < 16; k++) {
            
            global_print("uc", (u32)param->L[k], ' ');
        }
        u32 off = 0;

        for(u32 k = 0; k < 16; k++) {
            
            u32 l = (u32)param->L[k];
            /*
            if(l == 0) continue;
            for(u32 j = 0; j < l; j++) {
                if(j % 24 == 0) {
                    global_print("sus", "\n  L[", k, "] ");
                }
                global_print("uc", (u32)param->V[off + j], ' ');
            }
            */
            off += l;
        }

        i += off + sizeof(JPEGHuffmanParameter);
    }
    global_print("c", '\n');
    mem = (byte*)huff->params + len;
}
byte* PrintJPEGTables(byte* base, byte* mem) {

    for(bool run = true;run;) {

        JPEGMarker* marker = GetMarker(mem);
        ASSERT(marker->signature = 0xFF);
        mem = (byte*)marker;

        switch(marker->type) {
        case DQT:
            {              
                mem = PrintJPEGQT(base, mem);
                break;
            }
        case DHT:
            {
                mem = PrintJPEGHuffmanTable(base, mem);
                break;
            }
        case DAC:
            {
                auto arit = (JPEGArithmeticCondTable*)marker;
                u32 len = ReverseByteOrder(arit->len) - 2;
                global_print("xsuc", mem - base, " Arithmetic coding condition table\n ", len);

                u32 count = len / sizeof(JPEGArithmeticParameter);
                for(u32 i = 0; i < count; i++) {
                    
                    global_print("sucu sucu sucu",
                        "\n  table classe ", i, '\t', (u32)arit->params[i].tableClass,
                        "\n  arithmetic destination identifier ", i, '\t', (u32)arit->params[i].aritDstIdentifier,
                        "\n  conditioning table valuee ", i, '\t', (u32)arit->params[i].conditioningTableValue
                    );
                }
                global_print("c", '\n');
                mem = ((byte*)arit->params) + len;
                break;
            }
        case DRI:
            {
                auto restart = (JPEGRestartInterval*)marker;
                u32 len = ReverseByteOrder(restart->len) - 2;
                global_print("xsucsuc", mem - base, " Restart interval\n ", len , "\n restart interval ", (u32)ReverseByteOrder(restart->restartInterval), '\n');
                mem = (byte*)(restart + 1);
                break;
            }
        case COM:
            {
                auto com = (JPEGCommentSegment*)marker;
                u32 len = ReverseByteOrder(com->len) - 2;
                global_print("xsuss*c", mem - base, " Comment \n ", len , "\n ", com->comment, len, '\n');
                mem = com->comment + len;
                break;
            }
        case APP0:
            {
                auto app = (JPEGMarkerSegment*)marker;
                auto info = (APP0Payload*)app->payload;
                u32 len = ReverseByteOrder(app->len) - 2;
                global_print("xsuc", mem - base, " App \n len ", len);
                global_print("ss", "\n identifier ", info->identifier);
                global_print("su", "\n major ", (u32)info->major);
                global_print("su", "\n minor ", (u32)info->minor);
                global_print("su", "\n density unit ", (u32)info->densityUnit);
                global_print("su", "\n density x ", (u32)ReverseByteOrder(info->xDensity));
                global_print("su", "\n density y ", (u32)ReverseByteOrder(info->yDensity));
                if(info->xThumbnail) {
                    global_print("su", "\n thumbnail x ", (u32)ReverseByteOrder(info->xThumbnail));
                    global_print("su", "\n thumbnail y ", (u32)ReverseByteOrder(info->yThumbnail));
                }
                global_print("c", '\n');

                mem = app->payload + len;
                break;
            }
        default:
            run = false;
            break;
        }
    }

    global_io_flush();
    return mem;
}

byte* PrintJPEGFrameHeader(byte* begin, byte* mem) {

    JPEGFrameHeader* header = (JPEGFrameHeader*)GetMarker(mem);
    ASSERT(header->signature = 0xFF);
    
    if(header->type >= SOF0 && header->type <= SOF3) {
        const char* desc[] = {
            "(Baseline DCT)",
            "(Extended sequential DCT)",
            "(Progressive DCT)",
            "(Lossless sequential)"
        };
        u32 n = header->type - SOF0;
        global_print("xsuss", mem - begin, " SOF", n, desc[n], " non-differential, Huffman coding\n");
    }
    else if(header->type >= SOF5 && header->type <= SOF7) {
        const char* desc[] = {
            "(Differential sequential DCT)",
            "(Differential progressive DCT)",
            "(Differential lossless sequential)",
        };
        u32 n = header->type - SOF5;
        global_print("xsuss", mem - begin, " SOF", n, desc[n], " differential, Huffman coding\n");
    }
    else if(header->type >= SOF8 && header->type <= SOF11) {
        const char* desc[] = {
            "(Reserved for JPEG extensions)",
            "(Extended sequential DCT)",
            "(Progressive DCT)",
            "(Lossless sequential)",
        };
        u32 n = header->type - SOF8;
        global_print("xsuss", mem - begin, " SOF", n, desc[n], " non-differential, arithmetic coding\n");
    }
    else if(header->type >= SOF13 && header->type <= SOF15) {
        const char* desc[] = {
            "(Differential sequential DCT)",
            "(Differential progressive DCT)",
            "(Differential lossless sequential)",
        };
        u32 n = header->type - SOF13;
        global_print("xsuss", mem - begin, " SOF", n, desc[n], " differential, arithmetic coding\n");
    }
    else {
        ASSERT(false);
    }
    
    u32 len = ReverseByteOrder(header->len) - 2;
    u32 height = ReverseByteOrder(header->height);
    u32 width = ReverseByteOrder(header->width);
    u32 count = header->paramCount;
    u32 prec = header->samplePrecision;
    global_print("sususususu", " length ", len , "\n height ", height, "\n width ", width, "\n sameple precision ", prec, "\n parameter count ", count);
    
    
    u32 maxH = 0;
    u32 maxV = 0;
    for(u32 i = 0; i < count; i++) {
    	maxH = Max( (u32)header->params[i].samplingFactors.horizSamplingFactor, maxH);
    	maxV = Max( (u32)header->params[i].samplingFactors.vertSamplingFactor, maxV);
    }
    for(u32 i = 0; i < count; i++) {
    
    	f32 samplingX = (u32)header->params[i].samplingFactors.horizSamplingFactor;
    	f32 samplingY = (u32)header->params[i].samplingFactors.vertSamplingFactor;
    	
    	u32 dimX = (u32)ceil( (f32)width * (samplingX / (f32)maxH) );
    	u32 dimY = (u32)ceil( (f32)height * (samplingY / (f32)maxV) );
    	
        global_print("sucu sucu sucu sucu susus",
            "\n  component indentifier", i, '\t', (u32)header->params[i].componentIndentifier,
            "\n  QT destination selector", i, '\t', (u32)header->params[i].QTDstSelector,
            "\n  horizontal sampling Factor", i, '\t', (u32)header->params[i].samplingFactors.horizSamplingFactor,
            "\n  vertical sampling Factor", i, '\t', (u32)header->params[i].samplingFactors.vertSamplingFactor,
            "\n  component image dimensions (", dimY, " x ", dimX, ")\n"
        );
    }

    global_io_flush();
    return (byte*)(header->params + count);
}

byte* PrintJPEGScanHeader(byte* begin, byte* mem) {

    JPEGScanHeader* header = (JPEGScanHeader*)GetMarker(mem);
    ASSERT(header->signature == 0xFF && header->type == SOS);

    u32 len = ReverseByteOrder(header->len) - 2;
    u32 count = header->paramCount;
    global_print("xsusu", mem - begin, " SOS\n len ", len , "\n parameter count ", count);
    for(u32 i = 0; i < count; i++) {

        global_print("susu sucu sucuc",
            "\n  scan component selector", i, "\t\t\t", (u32)header->params[i].scanCompSelector,
            "\n  DC entropy coding table destination selector", i, '\t', (u32)header->params[i].selectors.DCentropyCodingTableDstSelector,
            "\n  AC entropy coding table destination selector", i, '\t', (u32)header->params[i].selectors.ACentropyCodingTableDstSelector, '\n'
        );
    }

    auto end = (JPEGScanHeaderEnd*)(header->params + count);
    global_print("su su su suc",
        "\n start of spectral selection ", (u32)end->startSpectralPredictorSelection,
        "\n end of spectral selection ", (u32)end->endSpectralSelection,
        "\n successive approximate bit position high ", (u32)end->bitPositions.approxBitPositionHigh,
        "\n successive approximate bit position low ", (u32)end->bitPositions.approxBitPositionLow, '\n'
    );

    global_io_flush();
    return (byte*)(end + 1);
}
byte* PrintJPEGScans(byte* begin, byte* mem) {

    JPEGMarker* marker;
    do {

        mem = PrintJPEGTables(begin, mem);
        mem = PrintJPEGScanHeader(begin, mem);
        mem = PrintJPEGTables(begin, mem);
        marker = GetMarker(mem);
        if(marker->type == DNL) {
            marker++;
            mem = (byte*)marker;
            global_print("xs", mem - begin, " DNL\n");
        }

    } while(marker->type == SOS);

    return mem;
}
void PrintJPEGMemory(byte* mem, u32 size) {
    
    const auto srcBegin = mem;
    if(!VerifyJFIFignature(mem)) {
        return;
    }

    mem = PrintJPEGTables(srcBegin, mem+2);
	auto marker = GetMarker(mem);
	if(marker->type == DHP) {
        global_print("xs", mem - srcBegin, " DHP\n");
	}

	do {

        mem = PrintJPEGTables(srcBegin, mem);
        mem = PrintJPEGFrameHeader(srcBegin, mem);
        mem = PrintJPEGScans(srcBegin, mem);

        marker = GetMarker(mem);
        mem = (byte*)marker;
	} while(marker->type != EOI);


    global_print("xs", mem - srcBegin, " EOI\n");
    global_io_flush();
}

byte* ParseJPEGTables(JPEGTables* tables, byte* mem) {

    for(bool run = true;run;) {

        JPEGMarker* marker = GetMarker(mem);
        ASSERT(marker->signature = 0xFF);
        mem = (byte*)marker;

        switch(marker->type) {
        case DQT:
            {
                auto qt = (JPEGQuantizationTable*)mem;
				u32 len = ReverseByteOrder(qt->len) - 2;
				for(u32 i = 0; i < len; ) {
					
					auto param = (JPEGQuantizationParameter*)((byte*)qt->params) + i;
					ASSERT((u32)param->dstIdentifier < 4);
					ASSERT(param->elemPrecision < 2);
					tables->quantizationTable[param->dstIdentifier] = param;
			        i += ((u32)param->elemPrecision + 1) * 64 + 1;
				}
				
				mem = (byte*)qt->params + len;
                break;
            }
        case DHT:
            {
			    auto huff = (JPEGHuffmanTable*)mem;
				u32 len = ReverseByteOrder(huff->len) - 2;

				for(u32 i = 0; i < len;) {

					auto param = (JPEGHuffmanParameter*)(((byte*)huff->params) + i);
					ASSERT(param->huffDstIdentifier < 4 && param->tableClass < 2);
                    tables->huffmanTables[param->tableClass * 4 + param->huffDstIdentifier] = param;

					u32 off = 0;
					for(u32 k = 0; k < 16; k++) {
						off += (u32)param->L[k];
					}

					i += (off + sizeof(JPEGHuffmanParameter));;
				}

				mem = (byte*)huff->params + len;            
                break;
            }
        case DAC:
            {
                auto arit = (JPEGArithmeticCondTable*)marker;
                u32 len = ReverseByteOrder(arit->len) - 2;

                u32 count = len / sizeof(JPEGArithmeticParameter);
                for(u32 i = 0; i < count; i++) {

                	auto param = arit->params + i;
					
					ASSERT(param->aritDstIdentifier < 4 && param->tableClass < 2);
                    tables->arithmeticTables[param->tableClass * 8 + param->aritDstIdentifier] = param;
               	}
                mem = (byte*)arit->params + len;
                break;
            }
        case DRI:
            {
                auto restart = (JPEGRestartInterval*)marker;
                u32 len = ReverseByteOrder(restart->len) - 2;
                mem = (byte*)(restart + 1);
                break;
            }
        case COM:
            {
                auto com = (JPEGCommentSegment*)marker;
                u32 len = ReverseByteOrder(com->len) - 2;
                mem = com->comment + len;
                break;
            }
        default:
            run = false;
            break;
        }
    }

    return mem;
}
u32 JPEGReceive(BitStream* stream, u32 n) {
    return ReadBitsJPEG(stream, n);
}
i32 JPEGExtend(i32 v, u32 t) {

    i32 vt = 1 << (t - 1);
    if(v < vt) {
        vt = (-1 << t) + 1;
        v += vt;
    }
    return v;
}
void DecodeJPEGACSeq(i16 block[63], BitStream* src, HuffmanDictionary* huff, u16 dequant[64]) {

    memset(block+1, 0, 63 * sizeof(i16));
    u32 k = 1;

    do {

        u32 rs = HuffmanDecodeJPEG(huff, src);

        u32 ssss = rs & 15;
        u32 r = rs >> 4;

        if(ssss == 0) {
            if(r != 15) {
                break;
            }
            k += 16;
        }
        else {
            k = k + r;
            u32 zig = jpeg_de_zig_zag[k++];
            i32 extend = JPEGExtend(JPEGReceive(src, ssss), ssss);
            block[zig] = (i32)dequant[zig] * extend;
        }

    } while(k < 64);
}
void stbi__idct_simd(u8* out, int out_stride, short data[64]);


i32 DecodeJPEGBlockHuffmanSeq(i16 dst[64], BitStream* src, HuffmanDictionary* huffDC, HuffmanDictionary* huffAC, u16 dequant[64], i32 pred) {
    
    u32 t = HuffmanDecodeJPEG(huffDC, src);
    u32 r = JPEGReceive(src, t);
    i32 diff = JPEGExtend(r, t);

    i32 dc = pred + diff;
    dst[0] = (i16)(dc * dequant[0]);

    DecodeJPEGACSeq(dst, src, huffAC, dequant);

    return dc;
}
i32 DecodeJPEGBlockHuffmanProgDC(i16 dst[64], BitStream* src, HuffmanDictionary* huffDC, u32 al, u32 ah, i32 pred) {

    i32 dc = pred;
    if(ah) {
        auto bit = ReadBitsJPEG(src, 1);
        dst[0] += bit << al;
    }
    else {
        ASSERT(al != 0);
        auto t = HuffmanDecodeJPEG(huffDC, src);
        i32 diff = 0;
        diff = JPEGExtend(JPEGReceive(src, t), t);
        dc = pred + diff;
        dst[0] = dc * (1 << al);
    }

    return dc;
}
i32 DecodeJPEGBlockHuffmanProgAC(i16 dst[64], BitStream* src, HuffmanDictionary* huffAC, u32 spectralStart, u32 spectralEnd, u32 al, u32 ah, i32 eobRun) {

    if (ah == 0) {

        auto shift = al;
        ASSERT(!eobRun);

        i32 k = al;
        do {

            int rs = HuffmanDecode(huffAC, src);
            i32 s = rs & 15;
            i32 r = rs >> 4;
            
            if (s == 0) {
                if (r < 15) {
                    eobRun = (1 << r);
                    if(r) {
                        eobRun += ReadBits(src, r);
                    }

                    eobRun--;
                    break;
                }
                k += 16;
            }
            else {

                k += r;
                u32 zig = jpeg_de_zig_zag[k++];
                auto extended = JPEGExtend(JPEGReceive(src, s), s);
                dst[zig] = (i16)(extended * (1 << shift));
            }

        } while (k <= spectralEnd);

    }
    else {
        // refinement scan for these AC coefficients
        short bit = (short) (1 << al);

        if(eobRun) {
            eobRun--;
            for(u32 k = spectralStart; k <= spectralEnd; k++) {

                i16* it = dst + jpeg_de_zig_zag[k];
                if(it[0] != 0) {

                    if(ReadBits(src, 1)) {
                        if( (it[0] & bit) == 0) {
                            if (it[0] > 0) {
                                it[0] += bit;
                            }
                            else {
                                it[0] -= bit;
                            }
                        }
                    }
                }
            }
        }
        else {
            
            u32 k = spectralStart;
            do {

                i32 r,s;
                i32 rs = HuffmanDecode(huffAC, src);
                s = rs & 15;
                r = rs >> 4;

                if (s == 0) {
                    if (r < 15) {
                        eobRun = (1 << r) - 1;
                        if (r) {
                            eobRun += ReadBits(src, r);
                        }

                        r = 64; // force end of block
                    }
                    else {
                        // r=15 s=0 should write 16 0s, so we just do
                        // a run of 15 0s and then write s (which is 0),
                        // so we don't have to do anything special here
                    }
                }
                else {

                    // sign bit
                    if (ReadBits(src, 1)) {
                        s = bit;
                    }
                    else {
                        s = -bit;
                    }
                }

                // advance by r
                while(k <= spectralEnd) {

                    i16* it = dst + jpeg_de_zig_zag[k++];
                    if (*it != 0) {

                        if(ReadBits(src, 1)) {

                            if( (*it & bit) == 0) {
                                if (*it > 0) {
                                    *it += bit;
                                }
                                else {
                                    *it -= bit;
                                }
                            }
                        }
                    }
                    else {
                        if (r == 0) {
                            *it = (i16)s;
                            break;
                        }
                        --r;
                    }
                }
            } while (k <= spectralEnd);
        }
    }

    return eobRun;
}

byte* DecodeJPEGEntropyHuffmanProgressive(JPEGFrameInfo* frameInfo, JPEGScanInfo* scanInfo, HuffmanDictionary* huffmans, i16* result[4]) {

    u32 DCpred[4]{};
    BitStream stream{scanInfo->entropy, 0, 0,0};

    auto dcHuff = huffmans + 0;
    auto acHuff = huffmans + 4;
    HuffmanDictionary* ac;
    HuffmanDictionary* dc;
    JPEGComponentInfo* compDescriptor;

    if(scanInfo->compCount == 1) {
        auto compIndex = scanInfo->comps[0].id - 1;
        auto mcuX = (frameInfo->comps[compIndex].width  + 7) >> 3;
        auto mcuY = (frameInfo->comps[compIndex].height + 7) >> 3;

        auto compSelect = &scanInfo->comps[0];
        compDescriptor = &frameInfo->comps[compSelect->id - 1];
        ASSERT(compDescriptor->id == compSelect->id);
        ac = acHuff + compSelect->entropyCodingSelectorAC;
        dc = dcHuff + compSelect->entropyCodingSelectorDC;
        u32 width = mcuX * compDescriptor->samplingFactorX * 8;
        u32 height = mcuY * compDescriptor->samplingFactorY * 8;
        u32 eobRun = 0;

        for(u32 i = 0; i < frameInfo->mcuY; i++) {
            for(u32 k = 0; k < frameInfo->mcuX; k++) {

                auto dst = result[0] + 64 * (i * mcuY + k);
                if(scanInfo->freqSpectrumBeginPredictor == 0) {
                    ASSERT(scanInfo->freqSpectrumEnd == 0);
                    DCpred[0] = DecodeJPEGBlockHuffmanProgDC(dst, &stream, dc, scanInfo->al, scanInfo->ah, DCpred[0]);
                }
                else {
                    eobRun = DecodeJPEGBlockHuffmanProgAC(dst, &stream, dc, scanInfo->freqSpectrumBeginPredictor, scanInfo->freqSpectrumEnd, scanInfo->al, scanInfo->ah, eobRun);
                }
            }
        }
    }
    else {

        for(u32 k = 0; k < frameInfo->mcuY; k++) {
            for(u32 i = 0; i < frameInfo->mcuX; i++) {

                for(u32 j = 0; j < scanInfo->compCount; j++) {

                    auto compSelect = &scanInfo->comps[j];
                    compDescriptor = &frameInfo->comps[j];
                    ASSERT(compDescriptor->id == compSelect->id);
                    ac = acHuff + compSelect->entropyCodingSelectorAC;
                    dc = dcHuff + compSelect->entropyCodingSelectorDC;
                    u32 width = frameInfo->mcuX * compDescriptor->samplingFactorX * 8;
                    u32 height = frameInfo->mcuY * compDescriptor->samplingFactorY * 8;

                    for(u32 y = 0; y < compDescriptor->samplingFactorY; y++) {
                        for(u32 x = 0; x < compDescriptor->samplingFactorX; x++) {

                            u32 imgCoordX = (i * compDescriptor->samplingFactorX + x) * 8;
                            u32 imgCoordY = (k * compDescriptor->samplingFactorY + y) * 8;
                            auto dst = result[j] + imgCoordY * width + imgCoordX;
                            DCpred[j] = DecodeJPEGBlockHuffmanProgDC(dst, &stream, dc, scanInfo->al, scanInfo->ah, DCpred[j]);
                        }
                    }
                }
            }
        }
    }
}
byte* DecodeJPEGEntropyHuffmanSequential(JPEGFrameInfo* frameInfo, JPEGScanInfo* scanInfo, HuffmanDictionary* huffmans, u16 QTs[4][64], u8* result[4]) {

    alignas (64) i16 block[64];
    u32 DCpred[4]{};
    BitStream stream{scanInfo->entropy, 0, 0,0};

    auto dcHuff = huffmans + 0;
    auto acHuff = huffmans + 4;
    HuffmanDictionary* ac;
    HuffmanDictionary* dc;
    JPEGComponentInfo* compDescriptor;

    u32 mcuY = frameInfo->mcuY;
    u32 mcuX = frameInfo->mcuX;
    if(scanInfo->compCount == 1) {
        auto compIndex = scanInfo->comps[0].id - 1;
        mcuX = (frameInfo->comps[compIndex].width  + 7) >> 3;
        mcuY = (frameInfo->comps[compIndex].height + 7) >> 3;
    }

    for(u32 k = 0; k < mcuY; k++) {
        for(u32 i = 0; i < mcuX; i++) {

            for(u32 j = 0; j < scanInfo->compCount; j++) {
                
                auto compSelect = &scanInfo->comps[j];
                compDescriptor = &frameInfo->comps[compSelect->id - 1];
                ASSERT(compDescriptor->id == compSelect->id);
                ac = acHuff + compSelect->entropyCodingSelectorAC;
                dc = dcHuff + compSelect->entropyCodingSelectorDC;
                u32 width = frameInfo->mcuX * compDescriptor->samplingFactorX * 8;
                u32 height = frameInfo->mcuY * compDescriptor->samplingFactorY * 8;

                u32 samplingFactorX = scanInfo->compCount == 1 ? 1 : compDescriptor->samplingFactorX;
                u32 samplingFactorY = scanInfo->compCount == 1 ? 1 : compDescriptor->samplingFactorY;
                
                for(u32 y = 0; y < samplingFactorY; y++) {
                    for(u32 x = 0; x < samplingFactorX; x++) {

                        u32 imgCoordX = (i * compDescriptor->samplingFactorX + x) * 8;
                        u32 imgCoordY = (k * compDescriptor->samplingFactorY + y) * 8;
                        DCpred[j] = DecodeJPEGBlockHuffmanSeq(block, &stream, dc, ac, QTs[compDescriptor->QTDstSelector], DCpred[j]);
                        //JPEGInverseDCT8x8(result[j] + width * imgCoordY + imgCoordX, block, width);
                        //stbi__idct_block(result[j] + width * imgCoordY + imgCoordX, width, block);
                        stbi__idct_simd(result[j] + width * imgCoordY + imgCoordX, width, block);
                    }
                }
            }
        }
    }

    return stream.bytePtr;
}
byte* ParseJPEGScan(JPEGScanInfo* info, byte* mem, LinearAllocator* alloc) {

    mem = ParseJPEGTables(&info->tables, mem);
    auto marker = GetMarker(mem);

    JPEGScanHeader* header = (JPEGScanHeader*)GetMarker(mem);
    ASSERT(header->signature == 0xFF && header->type == SOS);

    u32 len = ReverseByteOrder(header->len) - 2;
    info->compCount = header->paramCount;
    for(u32 i = 0; i < info->compCount; i++) {

        info->comps[i].id = header->params[i].scanCompSelector;
        info->comps[i].entropyCodingSelectorDC = header->params[i].selectors.DCentropyCodingTableDstSelector;
        info->comps[i].entropyCodingSelectorAC = header->params[i].selectors.ACentropyCodingTableDstSelector;
    }

    auto end = (JPEGScanHeaderEnd*)(header->params + info->compCount);
    info->entropy = (byte*)(end + 1);
    
    info->al = end->bitPositions.approxBitPositionLow;
    info->ah = end->bitPositions.approxBitPositionHigh;
    info->freqSpectrumBeginPredictor = end->startSpectralPredictorSelection;
    info->freqSpectrumEnd = end->endSpectralSelection;
  
    return info->entropy;
}
bool MatchMarker(JPEGMarker* marker, MarkerType* t, u32 count) {

    for(u32 i = 0; i < count; i++) {

        if (marker->type == t[i]) {
            return true;
        }
    }
    return false;
}
byte* ParseJPEGFrame(JPEGFrameInfo* info, byte* mem, u32 size, LinearAllocator* alloc) {

    JPEGTables tables{};
	mem = ParseJPEGTables(&tables, mem);

	auto marker = GetMarker(mem);
    info->hiearachical = false;
	if(marker->type == DHP) {
		info->hiearachical = true;
	}

    JPEGFrameHeader* header = (JPEGFrameHeader*)GetMarker(mem);
    ASSERT(header->signature = 0xFF);
    info->segment = header;

    
    if(header->type >= SOF0 && header->type <= SOF3) {
        JPEGMode modes[] = {
            JPEG_BASELINE_SEQUINTAL_DCT,
            JPEG_EXTENDEND_SEQUINTAL_DCT,
            JPEG_PROGRESSIVE_DCT,
            JPEG_LOSSLESS,
        };
        info->mode = modes[header->type - SOF0];
    }
    else if(header->type >= SOF5 && header->type <= SOF7) {
        header->type - SOF5;
    }
    else if(header->type >= SOF8 && header->type <= SOF11) {
        header->type - SOF8;
    }
    else if(header->type >= SOF13 && header->type <= SOF15) {
        header->type - SOF13;
    }
    else {
        ASSERT(false);
    }
    
    u32 len = ReverseByteOrder(header->len) - 2;
    info->height = ReverseByteOrder(header->height);
    info->width = ReverseByteOrder(header->width);
    u32 count = header->paramCount;
    info->samplePrecision = header->samplePrecision;
    
    u32 maxH = 0;
    u32 maxV = 0;
    for(u32 i = 0; i < count; i++) {
    	maxH = Max( (u32)header->params[i].samplingFactors.horizSamplingFactor, maxH);
    	maxV = Max( (u32)header->params[i].samplingFactors.vertSamplingFactor, maxV);
    }
    for(u32 i = 0; i < count; i++) {

        info->comps[i].id = header->params[i].componentIndentifier;
        info->comps[i].QTDstSelector = header->params[i].QTDstSelector;
        info->comps[i].samplingFactorX = header->params[i].samplingFactors.horizSamplingFactor;
        info->comps[i].samplingFactorY = header->params[i].samplingFactors.vertSamplingFactor;
    
    	f32 samplingX = (u32)header->params[i].samplingFactors.horizSamplingFactor;
    	f32 samplingY = (u32)header->params[i].samplingFactors.vertSamplingFactor;
    	
    	u32 dimX = (u32)ceil( (f32)info->width * (samplingX / (f32)maxH) );
    	u32 dimY = (u32)ceil( (f32)info->height * (samplingY / (f32)maxV) );
        info->comps[i].width = dimX;
        info->comps[i].height = dimY;
    }

    info->mcuX = (info->width + (maxV * 8) - 1) / (maxV * 8);
    info->mcuY = (info->height + (maxH * 8) - 1) / (maxH * 8);
    mem = (byte*)(header->params + count);
   
    info->scanCount = 0;
    LocalList<JPEGScanInfo> dummy;
    dummy.item.tables = tables;
    LocalList<JPEGScanInfo>* scans = &dummy;

    MarkerType types[] = {DQT,DHT,DAC,DRI,COM,SOS};
    do {

        LocalList<JPEGScanInfo>* tmp;
        ALLOCATE_LOCAL_LIST(tmp);
        tmp->next = scans;
        tmp->item.tables = scans->item.tables;
        scans = tmp;
        info->scanCount++;

        mem = ParseJPEGScan(&scans->item, mem, alloc);
        // entropy data
        marker = GetMarker(mem);
        if(marker->type == DNL) {
            marker++;
            mem = (byte*)marker;
            ASSERT(false);
        }

	} while(MatchMarker(marker, types, SIZE_OF_ARRAY(types)));

    info->scans = (JPEGScanInfo*)linear_allocate(alloc, info->scanCount * sizeof(JPEGScanInfo));
    for(u32 i = 0; i < info->scanCount && scans; i++) {
        info->scans[info->scanCount - (i+1)] = scans->item;
        scans = scans->next;
    }

	return (byte*)marker;
}
JPEGInfo ParseJPEGMemory(byte* mem, u32 size, LinearAllocator* alloc) {

    const auto srcBegin = mem;
    APP0Payload* header;
    if(!(header = VerifyJFIFignature(mem))) {
        return {};
    }
    auto seg = (JPEGMarkerSegment*)(mem + 2);
    mem = seg->payload + (ReverseByteOrder(seg->len) - 2);

	JPEGInfo info{};
    info.src = srcBegin;

    LocalList<JPEGFrameInfo>* frames = nullptr;

	do {

        LocalList<JPEGFrameInfo>* tmp;
        ALLOCATE_LOCAL_LIST(tmp);
        tmp->next = frames;
        frames = tmp;
        info.frameCount++;

        mem = ParseJPEGFrame(&tmp->item, mem, size, alloc);
	} while(GetMarker(mem)->type != EOI);

    info.frames = (JPEGFrameInfo*)linear_allocate(alloc, info.frameCount * sizeof(JPEGFrameInfo));
    for(u32 i = 0; i < info.frameCount && frames; i++) {
        info.frames[info.frameCount - (i+1)] = frames->item;
        frames = frames->next;
    }

    return info;
}


#define stbi__float2fixed(x)  (((int) ((x) * 4096.0f + 0.5f)) << 8)
u8 Sample2x2(u8* src, f32 u, f32 v, u32 stride, u32 w, u32 h) {

    f32 floorU = Max((f32)0, floor(u));
    f32 floorV = Max((f32)0, floor(v));

    f32 ceilU = Min((f32)w, ceil(u));
    f32 ceilV = Min((f32)h, ceil(v));

    u32 i0 = (floorV * (f32)stride) + floorU;
    u32 i1 = (ceilV * (f32)stride) + floorU;

    u32 i2 = (floorV * (f32)stride) + ceilU;
    u32 i3 = (ceilV * (f32)stride) + ceilU;

    auto weightX = u - floorU;
    auto weightY = v - floorV;

    f32 s0 = (1.f - weightX) * (f32)src[i0] + weightX * (f32)src[i1];
    f32 s1 = (1.f - weightX) * (f32)src[i2] + weightX * (f32)src[i3];

    return u8((1.f - weightY) * s0 + weightY * s1);
}
void JPEGResample2(JPEGFrameInfo* info, u8* dst, u8* Y, u8* Cb, u8* Cr) {

    Timer<micro_second_t> t;
    auto YsampleWidth = info->mcuX * info->comps[0].samplingFactorX * 8;
    auto CBsampleWidth = info->mcuX * info->comps[1].samplingFactorX * 8;
    auto CRsampleWidth = info->mcuX * info->comps[2].samplingFactorX * 8;

    auto maxH = Max(Max(info->comps[0].samplingFactorY, info->comps[1].samplingFactorY), info->comps[2].samplingFactorY);
    auto maxV = Max(Max(info->comps[0].samplingFactorX, info->comps[1].samplingFactorX), info->comps[2].samplingFactorX);

    f32 scaleX0 = (f32)maxH / (f32)info->comps[0].samplingFactorX;
    f32 scaleY0 = (f32)maxV / (f32)info->comps[0].samplingFactorY;

    f32 scaleX1 = (f32)maxH / (f32)info->comps[1].samplingFactorX;
    f32 scaleY1 = (f32)maxV / (f32)info->comps[1].samplingFactorY;

    f32 scaleX2 = (f32)maxH / (f32)info->comps[2].samplingFactorX;
    f32 scaleY2 = (f32)maxV / (f32)info->comps[2].samplingFactorY;

    auto maxIndex0 = YsampleWidth * info->comps[0].height + 1;
    auto maxIndex1 = CBsampleWidth * info->comps[1].height + 1;
    auto maxIndex2 = CRsampleWidth * info->comps[2].height + 1;

    for(u32 i = 0; i < info->height; i++) {
        for(u32 k = 0; k < info->width; k++) {

            auto x0 = (f32)k / scaleX0;
            auto y0 = (f32)i / scaleY0;

            auto x1 = (f32)k / scaleX1;
            auto y1 = (f32)i / scaleY1;

            auto x2 = (f32)k / scaleX2;
            auto y2 = (f32)i / scaleY2;

            u32 coordLin0 = (u32)y0 * YsampleWidth  + (u32)x0;
            u32 coordLin1 = (u32)y1 * CBsampleWidth + (u32)x1;
            u32 coordLin2 = (u32)y2 * CRsampleWidth + (u32)x2;

            u32 index0 = coordLin0;
            u32 index1 = coordLin1;
            u32 index2 = coordLin2;

            auto y = (i32)Y[index0];
            auto cb = (i32)Cb[index1] - 128;
            auto cr = (i32)Cr[index2] - 128;

            i32 r,g,b;
            int y_fixed = (y << 20) + (1<<19); // rounding
            r = y_fixed +  cr* stbi__float2fixed(1.40200f);
            g = y_fixed + (cr*-stbi__float2fixed(0.71414f)) + ((cb*-stbi__float2fixed(0.34414f)) & 0xffff0000);
            b = y_fixed                                     +   cb* stbi__float2fixed(1.77200f);
            r >>= 20;
            g >>= 20;
            b >>= 20;
            if ((unsigned) r > 255) { if (r < 0) r = 0; else r = 255; }
            if ((unsigned) g > 255) { if (g < 0) g = 0; else g = 255; }
            if ((unsigned) b > 255) { if (b < 0) b = 0; else b = 255; }

            dst[0] = (u8)r;
            dst[1] = (u8)g;
            dst[2] = (u8)b;
            dst[3] = 255;
            dst += 4;
        }
    }
}
void JPEGResampleSIMD(JPEGFrameInfo* info, u8* dst, u8* Y, u8* Cb, u8* Cr) {

    Timer<micro_second_t> t;
    auto YsampleWidth = info->mcuX * info->comps[0].samplingFactorX * 8;
    auto CBsampleWidth = info->mcuX * info->comps[1].samplingFactorX * 8;
    auto CRsampleWidth = info->mcuX * info->comps[2].samplingFactorX * 8;

    auto maxH = Max(Max(info->comps[0].samplingFactorY, info->comps[1].samplingFactorY), info->comps[2].samplingFactorY);
    auto maxV = Max(Max(info->comps[0].samplingFactorX, info->comps[1].samplingFactorX), info->comps[2].samplingFactorX);

    u32 stepX0 = (info->comps[0].samplingFactorX << 4) / maxH;
    u32 stepY0 = (info->comps[0].samplingFactorY << 4) / maxV;

    u32 stepX1 = (info->comps[1].samplingFactorX << 4) / maxH;
    u32 stepY1 = (info->comps[1].samplingFactorY << 4) / maxV;

    u32 stepX2 = (info->comps[2].samplingFactorX << 4) / maxH;
    u32 stepY2 = (info->comps[2].samplingFactorY << 4) / maxV;

    u32 index0_0 = 0;
    u32 index0_1 = 0;
    u32 index0_2 = 0;
    u32 index0_3 = 0;

    u32 index1_0 = 0;
    u32 index1_1 = 0;
    u32 index1_2 = 0;
    u32 index1_3 = 0;

    u32 index2_0 = 0;
    u32 index2_1 = 0;
    u32 index2_2 = 0;
    u32 index2_3 = 0;

    u32 y0_0 = 0;
    u32 y0_1 = 0;
    u32 y0_2 = 0;
    u32 y0_3 = 0;

    u32 y1_0 = 0;
    u32 y1_1 = 0;
    u32 y1_2 = 0;
    u32 y1_3 = 0;

    u32 y2_0 = 0;
    u32 y2_1 = 0;
    u32 y2_2 = 0;
    u32 y2_3 = 0;

    u32 x0_0 = stepX0 * 0;
    u32 x0_1 = stepX0 * 1;
    u32 x0_2 = stepX0 * 2;
    u32 x0_3 = stepX0 * 3;

    u32 x1_0 = stepX1 * 0;
    u32 x1_1 = stepX1 * 1;
    u32 x1_2 = stepX1 * 2;
    u32 x1_3 = stepX1 * 3;
    
    u32 x2_0 = stepX2 * 0;
    u32 x2_1 = stepX2 * 1;
    u32 x2_2 = stepX2 * 2;
    u32 x2_3 = stepX2 * 3;
    
    for(u32 i = 0, k = 3; i < (info->height * info->width) / 4; i++) {

        auto l0 = (i32)Y[index0_0 + (x0_0 >> 4)];
        auto l1 = (i32)Y[index0_1 + (x0_1 >> 4)];
        auto l2 = (i32)Y[index0_2 + (x0_2 >> 4)];
        auto l3 = (i32)Y[index0_3 + (x0_3 >> 4)];

        auto cb0 = (i32)Cb[index1_0 + (x1_0 >> 4)] - 128;
        auto cb1 = (i32)Cb[index1_1 + (x1_1 >> 4)] - 128;
        auto cb2 = (i32)Cb[index1_2 + (x1_2 >> 4)] - 128;
        auto cb3 = (i32)Cb[index1_3 + (x1_3 >> 4)] - 128;

        auto cr0 = (i32)Cr[index2_0 + (x2_0 >> 4)] - 128;
        auto cr1 = (i32)Cr[index2_1 + (x2_1 >> 4)] - 128;
        auto cr2 = (i32)Cr[index2_2 + (x2_2 >> 4)] - 128;
        auto cr3 = (i32)Cr[index2_3 + (x2_3 >> 4)] - 128;

        i32 r0,g0,b0;
        i32 r1,g1,b1;
        i32 r2,g2,b2;
        i32 r3,g3,b3;

        int y_fixed0 = (l0 << 20) + (1<<19); // rounding
        int y_fixed1 = (l1 << 20) + (1<<19);
        int y_fixed2 = (l2 << 20) + (1<<19);
        int y_fixed3 = (l3 << 20) + (1<<19);

        r0 = y_fixed0 +  cr0 * stbi__float2fixed(1.40200f);
        r1 = y_fixed1 +  cr1 * stbi__float2fixed(1.40200f);
        r2 = y_fixed2 +  cr2 * stbi__float2fixed(1.40200f);
        r3 = y_fixed3 +  cr3 * stbi__float2fixed(1.40200f);

        g0 = y_fixed0 + (cr0* -stbi__float2fixed(0.71414f)) + ((cb0 * -stbi__float2fixed(0.34414f)) & 0xffff0000);
        g1 = y_fixed1 + (cr1* -stbi__float2fixed(0.71414f)) + ((cb1 * -stbi__float2fixed(0.34414f)) & 0xffff0000);
        g2 = y_fixed2 + (cr2* -stbi__float2fixed(0.71414f)) + ((cb2 * -stbi__float2fixed(0.34414f)) & 0xffff0000);
        g3 = y_fixed3 + (cr3* -stbi__float2fixed(0.71414f)) + ((cb3 * -stbi__float2fixed(0.34414f)) & 0xffff0000);

        b0 = y_fixed0 + cb0 * stbi__float2fixed(1.77200f);
        b1 = y_fixed1 + cb1 * stbi__float2fixed(1.77200f);
        b2 = y_fixed2 + cb2 * stbi__float2fixed(1.77200f);
        b3 = y_fixed3 + cb3 * stbi__float2fixed(1.77200f);

        r0 >>= 20;
        r1 >>= 20;
        r2 >>= 20;
        r3 >>= 20;

        g0 >>= 20;
        g1 >>= 20;
        g2 >>= 20;
        g3 >>= 20;

        b0 >>= 20;
        b1 >>= 20;
        b2 >>= 20;
        b3 >>= 20;

        
        if ((unsigned) r0 > 255) { if (r0 < 0) r0 = 0; else r0 = 255; }
        if ((unsigned) r1 > 255) { if (r1 < 0) r1 = 0; else r1 = 255; }
        if ((unsigned) r2 > 255) { if (r2 < 0) r2 = 0; else r2 = 255; }
        if ((unsigned) r3 > 255) { if (r3 < 0) r3 = 0; else r3 = 255; }

        if ((unsigned) g0 > 255) { if (g0 < 0) g0 = 0; else g0 = 255; }
        if ((unsigned) g1 > 255) { if (g1 < 0) g1 = 0; else g1 = 255; }
        if ((unsigned) g2 > 255) { if (g2 < 0) g2 = 0; else g2 = 255; }
        if ((unsigned) g3 > 255) { if (g3 < 0) g3 = 0; else g3 = 255; }

        if ((unsigned) b0 > 255) { if (b0 < 0) b0 = 0; else b0 = 255; }
        if ((unsigned) b1 > 255) { if (b1 < 0) b1 = 0; else b1 = 255; }
        if ((unsigned) b2 > 255) { if (b2 < 0) b2 = 0; else b2 = 255; }
        if ((unsigned) b3 > 255) { if (b3 < 0) b3 = 0; else b3 = 255; }

        dst[0 ] = (u8)r0;
        dst[1 ] = (u8)g0;
        dst[2 ] = (u8)b0;
        dst[3 ] = 255;

        dst[4 ] = (u8)r1;
        dst[5 ] = (u8)g1;
        dst[6 ] = (u8)b1;
        dst[7 ] = 255;

        dst[8 ] = (u8)r2;
        dst[9 ] = (u8)g2;
        dst[10] = (u8)b2;
        dst[11] = 255;

        dst[12] = (u8)r3;
        dst[13] = (u8)g3;
        dst[14] = (u8)b3;
        dst[15] = 255;

        dst += 16;

        x0_0 += stepX0 * 4;
        x0_1 += stepX0 * 4;
        x0_2 += stepX0 * 4;
        x0_3 += stepX0 * 4;

        x1_0 += stepX1 * 4;
        x1_1 += stepX1 * 4;
        x1_2 += stepX1 * 4;
        x1_3 += stepX1 * 4;
        
        x2_0 += stepX2 * 4;
        x2_1 += stepX2 * 4;
        x2_2 += stepX2 * 4;
        x2_3 += stepX2 * 4;

        k += 4;
        if( k >= info->width) {

            u32 t = k - info->width;
            k = (t > 2) ? t : k;

            u32 lane0Mask0 = 0 - (x0_0 >= (info->comps[0].width << 4));
            u32 lane1Mask0 = 0 - (x0_1 >= (info->comps[0].width << 4));
            u32 lane2Mask0 = 0 - (x0_2 >= (info->comps[0].width << 4));
            u32 lane3Mask0 = 0 - (x0_3 >= (info->comps[0].width << 4));

            u32 lane0Mask1 = 0 - (x1_0 >= (info->comps[0].width << 3));
            u32 lane1Mask1 = 0 - (x1_1 >= (info->comps[0].width << 3));
            u32 lane2Mask1 = 0 - (x1_2 >= (info->comps[0].width << 3));
            u32 lane3Mask1 = 0 - (x1_3 >= (info->comps[0].width << 3));

            y0_0 += (stepY0 & lane0Mask0);
            y0_1 += (stepY0 & lane1Mask0);
            y0_2 += (stepY0 & lane2Mask0);
            y0_3 += (stepY0 & lane3Mask0);

            y1_0 += (stepY1 & lane0Mask1);
            y1_1 += (stepY1 & lane1Mask1);
            y1_2 += (stepY1 & lane2Mask1);
            y1_3 += (stepY1 & lane3Mask1);

            y2_0 += (stepY2 & lane0Mask1);
            y2_1 += (stepY2 & lane1Mask1);
            y2_2 += (stepY2 & lane2Mask1);
            y2_3 += (stepY2 & lane3Mask1);

            x0_0 -= (info->comps[0].width << 4) & lane0Mask0;
            x0_1 -= (info->comps[0].width << 4) & lane1Mask0;
            x0_2 -= (info->comps[0].width << 4) & lane2Mask0;
            x0_3 -= (info->comps[0].width << 4) & lane3Mask0;

            x1_0 -= (info->comps[0].width << 3) & lane0Mask1;
            x1_1 -= (info->comps[0].width << 3) & lane1Mask1;
            x1_2 -= (info->comps[0].width << 3) & lane2Mask1;
            x1_3 -= (info->comps[0].width << 3) & lane3Mask1;

            x2_0 -= (info->comps[0].width << 3) & lane0Mask1;
            x2_1 -= (info->comps[0].width << 3) & lane1Mask1;
            x2_2 -= (info->comps[0].width << 3) & lane2Mask1;
            x2_3 -= (info->comps[0].width << 3) & lane3Mask1;

            index0_0 = (y0_0 >> 4) * YsampleWidth;
            index0_1 = (y0_1 >> 4) * YsampleWidth;
            index0_2 = (y0_2 >> 4) * YsampleWidth;
            index0_3 = (y0_3 >> 4) * YsampleWidth;

            index1_0 = (y1_0 >> 4) * CBsampleWidth;
            index1_1 = (y1_1 >> 4) * CBsampleWidth;
            index1_2 = (y1_2 >> 4) * CBsampleWidth;
            index1_3 = (y1_3 >> 4) * CBsampleWidth;

            index2_0 = (y2_0 >> 4) * CRsampleWidth;
            index2_1 = (y2_1 >> 4) * CRsampleWidth;
            index2_2 = (y2_2 >> 4) * CRsampleWidth;
            index2_3 = (y2_3 >> 4) * CRsampleWidth;

        }
    }
}
void JPEGResampleUnrolled(JPEGFrameInfo* info, u8* dst, u8* Y, u8* Cb, u8* Cr) {

    Timer<micro_second_t> t;
    auto YsampleWidth = info->mcuX * info->comps[0].samplingFactorX * 8;
    auto CBsampleWidth = info->mcuX * info->comps[1].samplingFactorX * 8;
    auto CRsampleWidth = info->mcuX * info->comps[2].samplingFactorX * 8;

    auto maxH = Max(Max(info->comps[0].samplingFactorY, info->comps[1].samplingFactorY), info->comps[2].samplingFactorY);
    auto maxV = Max(Max(info->comps[0].samplingFactorX, info->comps[1].samplingFactorX), info->comps[2].samplingFactorX);

    u32 stepX0 = (info->comps[0].samplingFactorX << 4) / maxH;
    u32 stepY0 = (info->comps[0].samplingFactorY << 4) / maxV;

    u32 stepX1 = (info->comps[1].samplingFactorX << 4) / maxH;
    u32 stepY1 = (info->comps[1].samplingFactorY << 4) / maxV;

    u32 stepX2 = (info->comps[2].samplingFactorX << 4) / maxH;
    u32 stepY2 = (info->comps[2].samplingFactorY << 4) / maxV;

    u32 y0 = 0;
    u32 y1 = 0;
    u32 y2 = 0;

    u32 x0 = 0;
    u32 x1 = 0;
    u32 x2 = 0;

    u32 index0 = 0;
    u32 index1 = 0;
    u32 index2 = 0;

    for(u32 i = 0, k = 0; i < info->height * info->width; i++,k++) {

        auto y = (i32)Y[index0 + (x0 >> 4)];
        auto cb = (i32)Cb[index1 + (x1 >> 4)] - 128;
        auto cr = (i32)Cr[index2 + (x2 >> 4)] - 128;

        i32 r,g,b;
        int y_fixed = (y << 20) + (1<<19); // rounding
        r = y_fixed +  cr* stbi__float2fixed(1.40200f);
        g = y_fixed + (cr*-stbi__float2fixed(0.71414f)) + ((cb*-stbi__float2fixed(0.34414f)) & 0xffff0000);
        b = y_fixed                                     +   cb* stbi__float2fixed(1.77200f);
        r >>= 20;
        g >>= 20;
        b >>= 20;

        if ((unsigned) r > 255) { if (r < 0) r = 0; else r = 255; }
        if ((unsigned) g > 255) { if (g < 0) g = 0; else g = 255; }
        if ((unsigned) b > 255) { if (b < 0) b = 0; else b = 255; }

        dst[0] = (u8)r;
        dst[1] = (u8)g;
        dst[2] = (u8)b;
        dst[3] = 255;
        dst += 4;

        x0 += stepX0;
        x1 += stepX1;
        x2 += stepX2;

        if(k == info->width) {
            k = 0;
            y0 += stepY0;
            y1 += stepY1;
            y2 += stepY2;
            index0 = (y0 >> 4) * YsampleWidth;
            index1 = (y1 >> 4) * CBsampleWidth;
            index2 = (y2 >> 4) * CRsampleWidth;

            x0 = 0;
            x1 = 0;
            x2 = 0;
        }
    }
}
void JPEGResample(JPEGFrameInfo* info, u8* dst, u8* Y, u8* Cb, u8* Cr) {

    Timer<micro_second_t> t;
    auto YsampleWidth = info->mcuX * info->comps[0].samplingFactorX * 8;
    auto CBsampleWidth = info->mcuX * info->comps[1].samplingFactorX * 8;
    auto CRsampleWidth = info->mcuX * info->comps[2].samplingFactorX * 8;

    auto maxH = Max(Max(info->comps[0].samplingFactorY, info->comps[1].samplingFactorY), info->comps[2].samplingFactorY);
    auto maxV = Max(Max(info->comps[0].samplingFactorX, info->comps[1].samplingFactorX), info->comps[2].samplingFactorX);

    u32 stepX0 = (info->comps[0].samplingFactorX << 4) / maxH;
    u32 stepY0 = (info->comps[0].samplingFactorY << 4) / maxV;

    u32 stepX1 = (info->comps[1].samplingFactorX << 4) / maxH;
    u32 stepY1 = (info->comps[1].samplingFactorY << 4) / maxV;

    u32 stepX2 = (info->comps[2].samplingFactorX << 4) / maxH;
    u32 stepY2 = (info->comps[2].samplingFactorY << 4) / maxV;

    u32 y0 = 0;
    u32 y1 = 0;
    u32 y2 = 0;

    for(u32 i = 0; i < info->height; i++) {

        u32 index0 = (y0 >> 4) * YsampleWidth;
        u32 index1 = (y1 >> 4) * CBsampleWidth;
        u32 index2 = (y2 >> 4) * CRsampleWidth;

        u32 x0 = 0;
        u32 x1 = 0;
        u32 x2 = 0;
        for(u32 k = 0; k < info->width; k++) {

            auto y = (i32)Y[index0 + (x0 >> 4)];
            auto cb = (i32)Cb[index1 + (x1 >> 4)] - 128;
            auto cr = (i32)Cr[index2 + (x2 >> 4)] - 128;

            i32 r,g,b;
            int y_fixed = (y << 20) + (1<<19); // rounding
            r = y_fixed +  cr* stbi__float2fixed(1.40200f);
            g = y_fixed + (cr*-stbi__float2fixed(0.71414f)) + ((cb*-stbi__float2fixed(0.34414f)) & 0xffff0000);
            b = y_fixed                                     +   cb* stbi__float2fixed(1.77200f);
            r >>= 20;
            g >>= 20;
            b >>= 20;

            if ((unsigned) r > 255) { if (r < 0) r = 0; else r = 255; }
            if ((unsigned) g > 255) { if (g < 0) g = 0; else g = 255; }
            if ((unsigned) b > 255) { if (b < 0) b = 0; else b = 255; }

            dst[0] = (u8)r;
            dst[1] = (u8)g;
            dst[2] = (u8)b;
            dst[3] = 255;
            dst += 4;

            x0 += stepX0;
            x1 += stepX1;
            x2 += stepX2;
        }

        y0 += stepY0;
        y1 += stepY1;
        y2 += stepY2;
    }
}

u32 MaxMemoryRequiredJPEG(byte* mem) {

    for(auto marker = (JPEGMarker*)mem;; marker = GetMarker(mem)) {

        auto sof0 = (marker->type - SOF0) < 4;
        auto sof1 = (marker->type - SOF5) < 3;
        auto sof2 = (marker->type - SOF8) < 4;
        auto sof3 = (marker->type - SOF13) < 3;

        if(sof0 || sof1 || sof2 || sof3) {
            auto header = (JPEGFrameHeader*)marker;

            auto bits = header->samplePrecision / 8;
            auto w = ReverseByteOrder(header->width);
            auto h = ReverseByteOrder(header->height);

            u32 maxH = 0;
            u32 maxV = 0;
            for(u32 k = 0; k < header->paramCount; k++) {

                maxH = Max(maxH, (u32)header->params[k].samplingFactors.horizSamplingFactor);
                maxV = Max(maxV, (u32)header->params[k].samplingFactors.vertSamplingFactor);
            }
            auto mcuX = (w + (maxV * 8) - 1) / (maxV * 8);
            auto mcuY = (h + (maxH * 8) - 1) / (maxH * 8);
            
            u32 size = 0;
            for(u32 k = 0; k < header->paramCount; k++) {

                auto bits = header->samplePrecision >> 3;
                auto sampleWidth = mcuX * (u32)header->params[k].samplingFactors.horizSamplingFactor * 8;
                auto sampleHeight = mcuY * (u32)header->params[k].samplingFactors.vertSamplingFactor * 8;
                size += sampleWidth * sampleHeight * bits;
            }

            return size + w * h * 4;
        }

        mem = (byte*)(marker+1);
    }

    return ~u32(0);
}


ImageDescriptor LoadJPEG(byte* mem, u32 size, LinearAllocator* alloc) {

    alignas(64) i16 block[64];
    byte localMem[KILO_BYTE * 2];
    auto parserAlloc = make_linear_allocator(localMem, KILO_BYTE * 128);
    auto info = ParseJPEGMemory(mem, size, &parserAlloc);

    ImageDescriptor ret;
    ret.width = info.frames[0].width;
    ret.height = info.frames[0].height;
    auto imgSize = ret.width * ret.height * 4;
    u8* completeImg = (u8*)linear_allocate(alloc, imgSize);
    ret.img = (Pixel*)completeImg;
    const auto top0 = alloc->top;

    HuffmanDictionary huffs[8];
    for(u32 i = 0; i < 8; i++) {
        huffs[i] = AllocateHuffmanDict(alloc, 16);
    }

    u16 QTs[4][64];
    u8* imageComps[4];

    for(u32 i = 0; i < info.frameCount; i++) {

        const auto top = alloc->top;
        for(u32 k = 0; k < info.frames[i].segment->paramCount; k++) {

            ASSERT(info.frames[i].samplePrecision == 8);
            auto bits = info.frames[i].samplePrecision / 8;
            u32 sampleWidth = 0;
            u32 sampleHeight = 0;

            switch(info.frames[i].mode) {
            case JPEG_BASELINE_SEQUINTAL_DCT:
                sampleWidth = info.frames[i].mcuX * info.frames[i].comps[k].samplingFactorX * 8;
                sampleHeight = info.frames[i].mcuY * info.frames[i].comps[k].samplingFactorY * 8;
                break;
            case JPEG_PROGRESSIVE_DCT:
                bits = 16;
                sampleWidth = info.frames[i].comps[k].width;
                sampleHeight = info.frames[i].comps[k].height;
                break;
            }

            imageComps[k] = (u8*)linear_allocate(alloc, sampleWidth * sampleHeight * bits);
        }

        for(u32 k = 0; k < info.frames[i].scanCount; k++) {

            auto QT = info.frames[i].scans[k].tables.quantizationTable;
            for(u32 j = 0; j < 4; j++) {
                
                auto QQQ = QT[j];
                if(!QT[j]) continue;;
                for(u32 k = 0; k < 64; k++) {
                    u32 zig = jpeg_de_zig_zag[k];
                    if(QT[j]->elemPrecision) {
                        QTs[j][zig] = Mem<u16>(QT[j]->Q + k*2);
                    }
                    else {
                        QTs[j][zig] = QT[j]->Q[k];
                    }
                }
            }
            for(u32 j = 0; j < 8; j++) {

                auto huffmanDescriptor = info.frames[i].scans[k].tables.huffmanTables[j];
                if(!huffmanDescriptor) continue;
                ComputeHuffmanTableJPEG(huffs+j, huffmanDescriptor->L, huffmanDescriptor->V);
            }

            switch(info.frames[i].mode) {
            case JPEG_BASELINE_SEQUINTAL_DCT:
                DecodeJPEGEntropyHuffmanSequential(info.frames + i, info.frames[i].scans + k, huffs, QTs, imageComps);
                break;
            case JPEG_PROGRESSIVE_DCT:
                DecodeJPEGEntropyHuffmanProgressive(info.frames + i, info.frames[i].scans + k, huffs, (i16**)imageComps);
                break;
            }
        }

        if(info.frames[i].mode == JPEG_PROGRESSIVE_DCT) {

            u8* compFinal[4];
            for(u32 k = 0; k < info.frames[i].segment->paramCount; k++) {
                
                u32 w = (info.frames[i].comps[k].width + 7) >> 3;
                u32 h = (info.frames[i].comps[k].height + 7) >> 3;

                u32 sampleWidth = info.frames[i].mcuX * info.frames[i].comps[k].samplingFactorX * 8;
                u32 sampleHeight = info.frames[i].mcuY * info.frames[i].comps[k].samplingFactorY * 8;
                compFinal[k] = (u8*)linear_allocate(alloc, sampleWidth * sampleHeight);
                
                auto compCoef = (i16*)(imageComps[k]);
                auto qt = QTs[info.frames[i].comps[k].QTDstSelector];
                for(u32 j = 0; j < h; j++) {
                    for(u32 l = 0; l < w; l++) {
                        
                        i16* coefs = compCoef + 64 * ((j * info.frames[i].comps[k].width/8) + l);
                        for(u32 z = 0; z < 64; z++) {
                            block[z] = coefs[z] * qt[z];
                        }
                        stbi__idct_simd(compFinal[k], sampleWidth, block);
                    }
                }

                imageComps[k] = compFinal[k];
            }
        }

        JPEGResample(info.frames + i, completeImg, imageComps[0], imageComps[1], imageComps[2]);
        alloc->top = top;
    }

    alloc->top = top0;
    return ret;
}

u32 MemCmp(byte* s0, byte* s1, u32 size) {

    for(u32 i = 0; i < size; i++) {

        if(s0[i] != s1[i]) {
            return i;
        }
    }
    return size;
}

void SlowIDCT1D(f32* block, u32 n ) {

    f32 cpy[n];
    memcpy(cpy, block, n * sizeof(f32));
    for(u32 i = 0; i < 8; i++) {

        f32 sum = 0;
        for(u32 u = 0; u < 8; u++) {

            f32 alpha = 1;
            if(u == 0) {
                alpha = sqrt(1.0f / f32(n));
            }
            else {
                alpha = sqrt(2.0f / f32(n));
            }
            f32 cosArg = (3.14 * f32(u)) / (2.0f * f32(n));
            f32 c = cos(cosArg * (2.0f * f32(i) + 1.0f) );
            sum += alpha * c * cpy[u];
        }

        block[i] = sum;
    }
}
f32 IDCT2D(u32 x, u32 y, i16 data[64]) {

    f32 sum = 0;
    for(u32 u = 0; u < 8; u++) {

        f32 alphaU;
        if(u == 0) {
            alphaU = 1.f / sqrt(2.f);
        }
        else {
            alphaU = 1.f;
        }
        f32 cosU = cos( ((2.f * x + 1.f) * (3.14 * u)) / 16.f );
        for(u32 v = 0; v < 8; v++) {

            f32 alphaV;
            if(v == 0) {
                alphaV = 1.f / sqrt(2.f);
            }
            else {
                alphaV = 1.f;
            }

            f32 cosV = cos( ((2.f * y + 1.f) * (3.14 * v)) / 16.f );
            sum += alphaU * alphaV * (f32)data[u * 8 + v] * cosU * cosV;
        }
    }

    return sum / 4.f;
}
void SlowComputeJPEGIDCT2D(u8* dst, i16 src[64], u32 stride) {

    //i16 copy[64];
    //memcpy(copy, src, sizeof(copy));
    for(u32 x = 0; x < 8; x++) {

        for(u32 y = 0; y < 8; y++) {
            dst[y] = (u8) (round(IDCT2D(x, y, src)) + 128);
        }
        dst += stride;
    }
}

void SlowIDCT(f32 block[64]) {


    f32 col[8];
    for(u32 i = 0; i < 8; i++) {

        col[0] = block[8 * 0 + i];
        col[1] = block[8 * 1 + i];
        col[2] = block[8 * 2 + i];
        col[3] = block[8 * 3 + i];
        col[4] = block[8 * 4 + i];
        col[5] = block[8 * 5 + i];
        col[6] = block[8 * 6 + i];
        col[7] = block[8 * 7 + i];
        SlowIDCT1D(col, 8);
        block[8 * 0 + i] = col[0];
        block[8 * 1 + i] = col[1];
        block[8 * 2 + i] = col[2];
        block[8 * 3 + i] = col[3];
        block[8 * 4 + i] = col[4];
        block[8 * 5 + i] = col[5];
        block[8 * 6 + i] = col[6];
        block[8 * 7 + i] = col[7];
    }
    for(u32 i = 0; i < 8; i++) {
        SlowIDCT1D(block + i*8, 8);

        for(u32 k = 0; k < 8; k++) {
            block[i * 8 + k] = round(block[i * 8 + k] + 128);
        }
    }
}
void SlowIDCT2(f32 block[64]) {

    for(u32 i = 0; i < 8; i++) {
        SlowIDCT1D(block + i*8, 8);
    }
    f32 col[8];
    for(u32 i = 0; i < 8; i++) {

        col[0] = block[8 * 0 + i];
        col[1] = block[8 * 1 + i];
        col[2] = block[8 * 2 + i];
        col[3] = block[8 * 3 + i];
        col[4] = block[8 * 4 + i];
        col[5] = block[8 * 5 + i];
        col[6] = block[8 * 6 + i];
        col[7] = block[8 * 7 + i];
        SlowIDCT1D(col, 8);
        block[8 * 0 + i] = col[0];
        block[8 * 1 + i] = col[1];
        block[8 * 2 + i] = col[2];
        block[8 * 3 + i] = col[3];
        block[8 * 4 + i] = col[4];
        block[8 * 5 + i] = col[5];
        block[8 * 6 + i] = col[6];
        block[8 * 7 + i] = col[7];
    }
    for(u32 k = 0; k < 64; k++) {
        block[k] = round(block[k] + 128);
    }
}

void JPEGInverseDCT8x8(u8* dst, i16 src[64], u32 stride) {
    
    i16 col[8];
    i16 cpy[64];
    for(u32 i = 0; i < 8; i++) {

        col[0] = src[8 * 0 + i];
        col[1] = src[8 * 1 + i];
        col[2] = src[8 * 2 + i];
        col[3] = src[8 * 3 + i];
        col[4] = src[8 * 4 + i];
        col[5] = src[8 * 5 + i];
        col[6] = src[8 * 6 + i];
        col[7] = src[8 * 7 + i];
        FIDCT8(col, col);
        cpy[8 * 0 + i] = col[0];
        cpy[8 * 1 + i] = col[1];
        cpy[8 * 2 + i] = col[2];
        cpy[8 * 3 + i] = col[3];
        cpy[8 * 4 + i] = col[4];
        cpy[8 * 5 + i] = col[5];
        cpy[8 * 6 + i] = col[6];
        cpy[8 * 7 + i] = col[7];
    }
    for(u32 i = 0; i < 8; i++) {

        FIDCT8(cpy + i * 8, cpy + i * 8);
        for(u32 k = 0; k < 8; k++) {
            dst[k] = (u8)(cpy[i * 8 + k] + 128);
        }
        dst += stride;
    }
}

constexpr f64 S[] = {
	0.353553390593273762200422, // 1.0f / (2.0f * sqrt(2));
	0.254897789552079584470970, // 1.0f / (4.0f * cos(1.0f * PI / 16.0f) );
	0.270598050073098492199862, // 1.0f / (4.0f * cos(2.0f * PI / 16.0f) );
	0.300672443467522640271861, // 1.0f / (4.0f * cos(3.0f * PI / 16.0f) );
	0.353553390593273762200422, // 1.0f / (4.0f * cos(4.0f * PI / 16.0f) );
	0.449988111568207852319255, // 1.0f / (4.0f * cos(5.0f * PI / 16.0f) );
	0.653281482438188263928322, // 1.0f / (4.0f * cos(6.0f * PI / 16.0f) );
	1.281457723870753089398043, // 1.0f / (4.0f * cos(7.0f * PI / 16.0f) );
};
constexpr f64 A[] = {
	NAN,
	0.707106781186547524400844, // cos(4.0f * PI / 16.0f);
	0.541196100146196984399723, // cos(2.0f * PI / 16.0f) - cos(6.0f * PI / 16.0f);
	0.707106781186547524400844, // cos(4.0f * PI / 16.0f);
	1.306562964876376527856643, // cos(6.0f * PI / 16.0f) + cos(2.0f * PI / 16.0f);
	0.382683432365089771728460, // cos(6.0f * PI / 16.0f);
};


void FIDCT8(i16 dst[8], i16 src[8]) {

	const f32 v15 = src[0] / S[0];
	const f32 v26 = src[1] / S[1];
	const f32 v21 = src[2] / S[2];
	const f32 v28 = src[3] / S[3];
	const f32 v16 = src[4] / S[4];
	const f32 v25 = src[5] / S[5];
	const f32 v22 = src[6] / S[6];
	const f32 v27 = src[7] / S[7];
	
	const f32 v19 = (v25 - v28) / 2;
	const f32 v20 = (v26 - v27) / 2;
	const f32 v23 = (v26 + v27) / 2;
	const f32 v24 = (v25 + v28) / 2;
	
	const f32 v7  = (v23 + v24) / 2;
	const f32 v11 = (v21 + v22) / 2;
	const f32 v13 = (v23 - v24) / 2;
	const f32 v17 = (v21 - v22) / 2;
	
	const f32 v8 = (v15 + v16) / 2;
	const f32 v9 = (v15 - v16) / 2;
	
	const f32 v18 = (v19 - v20) * A[5];  // Different from original
	const f32 v12 = (v19 * A[4] - v18) / (A[2] * A[5] - A[2] * A[4] - A[4] * A[5]);
	const f32 v14 = (v18 - v20 * A[2]) / (A[2] * A[5] - A[2] * A[4] - A[4] * A[5]);
	
	const f32 v6 = v14 - v7;
	const f32 v5 = v13 / A[3] - v6;
	const f32 v4 = -v5 - v12;
	const f32 v10 = v17 / A[1] - v11;
	
	const f32 v0 = (v8 + v11) / 2;
	const f32 v1 = (v9 + v10) / 2;
	const f32 v2 = (v9 - v10) / 2;
	const f32 v3 = (v8 - v11) / 2;
	
	dst[0] = round((v0 + v7) / 2);
	dst[1] = round((v1 + v6) / 2);
	dst[2] = round((v2 + v5) / 2);
	dst[3] = round((v3 + v4) / 2);
	dst[4] = round((v3 - v4) / 2);
	dst[5] = round((v2 - v5) / 2);
	dst[6] = round((v1 - v6) / 2);
	dst[7] = round((v0 - v7) / 2);
}
/*
void FixedPointFastInverseDCT8(i16 dst[8], i16 src[8]) {

    constexpr f64 F64_MUL_12_BIT = 1 << 12;
    constexpr i32 S_i32[] = {
        i32( (1.0 / 0.353553390593273762200422) * F64_MUL_12_BIT), // 1.0f / (2.0f * sqrt(2));
        i32( (1.0 / 0.254897789552079584470970) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(1.0f * PI / 16.0f) );
        i32( (1.0 / 0.270598050073098492199862) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(2.0f * PI / 16.0f) );
        i32( (1.0 / 0.300672443467522640271861) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(3.0f * PI / 16.0f) );
        i32( (1.0 / 0.353553390593273762200422) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(4.0f * PI / 16.0f) );
        i32( (1.0 / 0.449988111568207852319255) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(5.0f * PI / 16.0f) );
        i32( (1.0 / 0.653281482438188263928322) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(6.0f * PI / 16.0f) );
        i32( (1.0 / 1.281457723870753089398043) * F64_MUL_12_BIT), // 1.0f / (4.0f * cos(7.0f * PI / 16.0f) );
    };
    constexpr i32 A_i32[] = {
        i32(0.707106781186547524400844 * F64_MUL_12_BIT), // cos(4.0f * PI / 16.0f);
        i32(0.541196100146196984399723 * F64_MUL_12_BIT), // cos(2.0f * PI / 16.0f) - cos(6.0f * PI / 16.0f);
        i32(0.707106781186547524400844 * F64_MUL_12_BIT), // cos(4.0f * PI / 16.0f);
        i32(1.306562964876376527856643 * F64_MUL_12_BIT), // cos(6.0f * PI / 16.0f) + cos(2.0f * PI / 16.0f);
        i32(0.382683432365089771728460 * F64_MUL_12_BIT), // cos(6.0f * PI / 16.0f);
    };

}
*/
#include <open_type_loader.h>
#include <math.h>
#include <interface.h>

u32 ComputeTableChecksumOTF(u32* table, u32 length) {

    u32 sum = 0L;
    u32* endptr = table + ( (length + 3) & ~3) / sizeof(u32);
    while (table < endptr) {
        sum += *table++;
    }

    u32 end = 0;
    u8* rem = (u8*)endptr;
    for(u32 i = 0; i < length & 3; i++) {
        end |= (u32)rem[i] << (3-i) * 8;
    }
    sum += end;

    return sum;
}

TableRecordOTF* GetTableOTF(OTFInfo* info, u32 tag, TableDirectoryOTF* dirs) {

    for(u32 i = 0; i < info->tableCount; i++) {
        u32 t = Mem<u32>(dirs->tableRecords[i].tableTag);
        if(tag == t) {

            return dirs->tableRecords + i;
        }
    }

    return nullptr;
}
u32 SearchGlyphIndexDenseMap(OTFDenseCodePointMap* map, u32 codePoint) {

    u32 mask = u32(0) - map->sequential;
    for(u32 i = 0; i < map->groupCount; i++) {
        if(codePoint <= map->groups[i].endCharCode) {

            if(codePoint < map->groups[i].startCharCode) {
                return 0;
            }
            return (codePoint - map->groups[i].startCharCode) & mask + map->groups[i].startGlyphID;
        }
    }
    return 0;
}
u32 SearchGlyphIndexSparseMap(OTFSparseCodePointMap* map, u32 codePoint) {

    u32 codePointSegment = 0;
    for(u32 i = 0; i < map->segmentCount; i++) {
        if(map->segments[i].end >= codePoint) {
            codePointSegment = i;
            break;
        }
    }
    if(map->segments[codePointSegment].start <= codePoint) {

        if(map->segments[codePointSegment].offset) {

            auto id = (map->segments[codePointSegment].offset >> 1) + (codePoint - map->segments[codePointSegment].start);
            auto glypID = map->glyphIDs[id];
            if(!glypID) return 0;

            return map->segments[codePointSegment].delta + glypID & 0xFFFF;
        }
        return (map->segments[codePointSegment].delta + codePoint) & 0xFFFF;
    }
    return 0;
}
u32 SearchGlyphIndex(byte* mem, u32 codePoint) {

    u16 format = Mem<u16>(mem);
    format = ReverseByteOrder(format);
    switch(format) {
    case 4:
        {
            auto subtable = (Format4OTFHeader*)mem;
            u32 segCount = ReverseByteOrder(subtable->segCountX2) / 2;
            auto body0 = (Format4OTFBody0*)(subtable->endCode    + (segCount - 0));
            auto body1 = (Format4OTFBody1*)(body0->startCode     + (segCount - 1));
            auto body2 = (Format4OTFBody2*)(body1->lastStartCode + (segCount - 1));
            auto body3 = (Format4OTFBody3*)(body1->idDelta       + (segCount - 1));

            auto startCodes = body0->startCode;
            auto endCodes = subtable->endCode;
            auto IDDeltas = body1->idDelta;
            auto IDRangeOffsets = body2->idRangeOffsets;
            auto glyphIDs = body3->glyphID;

            u32 codePointSegment = 0;
            for(u32 i = 0; i < segCount; i++) {
                if(endCodes[i] >= codePoint) {
                    codePointSegment = i;
                    break;
                }
            }
            if(startCodes[codePointSegment] <= codePoint) {

                if(IDRangeOffsets[codePointSegment]) {
                    auto id = (IDRangeOffsets[codePointSegment] >> 1) + (codePoint - startCodes[codePointSegment]);
                    auto glypID = glyphIDs[id];
                    if(!glypID) return 0;

                    return (IDDeltas[codePointSegment] + glypID) & 0xFFFF;
                }
                return (IDDeltas[codePointSegment] + codePoint) & 0xFFFF;
            }
            return 0;
        }
    case 6:
        {
            auto subtable = (Format6OTF*)mem;
            if(subtable->firstCode > codePoint || (codePoint - subtable->firstCode) > (subtable->entryCount-1)) return 0;
            return subtable->glyphIDs[codePoint - subtable->firstCode];
        }
    case 10:
        {
            auto subtable = (Format10OTF*)mem;
            if(subtable->startCharCode > codePoint || (codePoint - subtable->startCharCode) > (subtable->numChars-1)) return 0;
            return subtable->glyphIDs[codePoint - subtable->startCharCode];
        }
    case 12:
        {
            auto subtable = (Format12OTF*)mem;
            for(u32 i = 0; i < subtable->numGroups; i++) {
                if(codePoint <= subtable->groups[i].endCharCode) {

                    if(codePoint < subtable->groups[i].startCharCode) {
                        return 0;
                    }
                    return (codePoint - subtable->groups[i].startCharCode)  + subtable->groups[i].startGlyphID;
                }
            }
            return 0;
        }
    case 13:
        {
            auto subtable = (Format13OTF*)mem;
            for(u32 i = 0; i < subtable->numGroups; i++) {
                if(codePoint <= subtable->groups[i].endCharCode) {

                    if(codePoint < subtable->groups[i].startCharCode) {
                        return 0;
                    }
                    return subtable->groups[i].glyphID;
                }
            }
            return 0;
        }
    case 14:
        {
            auto subtable = (Format14OTF*)mem;
            ASSERT(false);
            break;
        }
    }
}

void ParseUnicodePlatformSubTableOTF(OTFInfo* info, byte* mem, u32 encoding, LinearAllocator* alloc) {

    u16 format = Mem<u16>(mem);
    format = ReverseByteOrder(format);
    switch(format) {
    case 4:
        {
            auto subtable = (Format4OTFHeader*)mem;
            auto subtableEnd = mem + subtable->length;
            u32 segCount = ReverseByteOrder(subtable->segCountX2) / 2;
            auto body0 = (Format4OTFBody0*)(subtable->endCode       + (segCount - 0));
            auto body1 = (Format4OTFBody1*)(body0->startCode        + (segCount - 1));
            auto body2 = (Format4OTFBody2*)(body1->idDelta          + (segCount - 1));
            auto body3 = (Format4OTFBody3*)(body2->idRangeOffsets   + (segCount - 1));

            info->sparse = 1;
            auto glyphIDCount = (u16*)subtableEnd - body3->glyphID;
            info->sparseMap.segmentCount = segCount;
            info->sparseMap.segments = (OTFCodePointSegment*)linear_allocate(alloc, segCount * sizeof(OTFCodePointSegment));

            for(u32 i = 0; i < segCount; i++) {
                info->sparseMap.segments[i].start   = ReverseByteOrder(body0->startCode[i]);
                info->sparseMap.segments[i].end     = ReverseByteOrder(subtable->endCode[i]);
                info->sparseMap.segments[i].delta   = ReverseByteOrder(body1->idDelta[i]);
                info->sparseMap.segments[i].offset  = ReverseByteOrder(body2->idRangeOffsets[i]);
            }
            info->sparseMap.glyphIDCount = glyphIDCount;
            info->sparseMap.glyphIDs = (u16*)linear_allocate(alloc, glyphIDCount * sizeof(u16));
            for(u32 i = 0; i < glyphIDCount; i++) {
                info->sparseMap.glyphIDs[i] = ReverseByteOrder(body3->glyphID[i]);
            }

            break;
        }
    case 6:
        {
            auto subtable = (Format6OTF*)mem;
            info->sparse = 1;
            info->sparseMap.segmentCount = 1;
            info->sparseMap.segments = (OTFCodePointSegment*)linear_allocate(alloc, sizeof(OTFCodePointSegment));
            info->sparseMap.segments[0].start = ReverseByteOrder(subtable->firstCode);
            info->sparseMap.segments[0].end = info->sparseMap.segments[0].start + subtable->entryCount - 1;
            info->sparseMap.segments[0].offset = 0;
            info->sparseMap.segments[0].delta = 0;

            info->sparseMap.glyphIDCount = ReverseByteOrder(subtable->entryCount);
            info->sparseMap.glyphIDs = (u16*)linear_allocate(alloc, info->sparseMap.glyphIDCount * sizeof(u16));
            for(u32 i = 0; i < info->sparseMap.glyphIDCount; i++) {
                info->sparseMap.glyphIDs[i] = ReverseByteOrder(info->sparseMap.glyphIDs[i]);
            }
            break;
        }
    case 10:
        {
            auto subtable = (Format10OTF*)mem;
            info->sparse = 1;
            info->sparseMap.segmentCount = 1;
            info->sparseMap.segments = (OTFCodePointSegment*)linear_allocate(alloc, sizeof(OTFCodePointSegment));
            info->sparseMap.segments[0].start = ReverseByteOrder(subtable->startCharCode);
            info->sparseMap.segments[0].end = info->sparseMap.segments[0].start + subtable->numChars - 1;
            info->sparseMap.segments[0].offset = 0;
            info->sparseMap.segments[0].delta = 0;

            info->sparseMap.glyphIDCount = ReverseByteOrder(subtable->numChars);
            info->sparseMap.glyphIDs = (u16*)linear_allocate(alloc, info->sparseMap.glyphIDCount * sizeof(u16));
            for(u32 i = 0; i < info->sparseMap.glyphIDCount; i++) {
                info->sparseMap.glyphIDs[i] = ReverseByteOrder(subtable->glyphIDs[i]);
            }
            break;
        }
    case 12:
        {
            auto subtable = (Format12OTF*)mem;
            info->sparse = 0;
            info->denseMap.sequential = 1;
            info->denseMap.groupCount = ReverseByteOrder(subtable->numGroups);
            info->denseMap.groups = (SequentialMapGroupOTF*)linear_allocate(alloc, info->denseMap.groupCount * sizeof(SequentialMapGroupOTF));
            for(u32 i = 0; i < info->denseMap.groupCount; i++) {

                info->denseMap.groups[i].endCharCode    = ReverseByteOrder(subtable->groups[i].endCharCode);
                info->denseMap.groups[i].startCharCode  = ReverseByteOrder(subtable->groups[i].startCharCode);
                info->denseMap.groups[i].startGlyphID   = ReverseByteOrder(subtable->groups[i].startGlyphID);
            }
            break;
        }
    case 13:
        {
            auto subtable = (Format13OTF*)mem;
            info->sparse = 0;
            info->denseMap.sequential = 0;
            info->denseMap.groupCount = ReverseByteOrder(subtable->numGroups);
            info->denseMap.groups = (SequentialMapGroupOTF*)linear_allocate(alloc, info->denseMap.groupCount * sizeof(SequentialMapGroupOTF));
            for(u32 i = 0; i < info->denseMap.groupCount; i++) {

                info->denseMap.groups[i].endCharCode    = ReverseByteOrder(subtable->groups[i].endCharCode);
                info->denseMap.groups[i].startCharCode  = ReverseByteOrder(subtable->groups[i].startCharCode);
                info->denseMap.groups[i].startGlyphID   = ReverseByteOrder(subtable->groups[i].glyphID);
            }
            break;
        }
    case 14:
        {
            ASSERT(false);
            auto subtable = (Format14OTF*)mem;
            break;
        }
    }
}
u64 ReadBytes(u32 size, byte* src) {

    u64 v = 0;
    ASSERT(size < 5);
    for(u32 i = 0; i < size; i++) {
        v |= ((u64)src[i]) << (size - i - 1) * 8;
    }

    return v;
}

byte* GetEndofCFFIndex(IndexCFF* index) {

    auto count = ReverseByteOrder(index->count);
    ASSERT(count != 0);
    auto objData = index->offsets + index->offSize * (count+1);

    u32 endOffset = ReadBytes(index->offSize, index->offsets + count * index->offSize);
    return objData + endOffset - 1;
}

CFFIndexEntry GetIndexEntryCFF(IndexCFF* index, u32 i) {

    u32 count = ReverseByteOrder(index->count);
    ASSERT(i < count);
    byte* data = index->offsets + index->offSize * (count+1);

    u32 start = ReadBytes(index->offSize, index->offsets + index->offSize * (i + 0)) - 1;
    u32 end   = ReadBytes(index->offSize, index->offsets + index->offSize * (i + 1)) - 1;

    return CFFIndexEntry {
        .begin  = data + start,
        .end    = data + end,
    };
}
i64 GetIntCFF(u8* src) {

    if(!src) return (i64(1) << 63);

    u8 b0 = src[0];
    if(b0 >= 32 && b0 <= 246) {
        src += 1;
        return (i64)b0 - 139;
    }
    if(b0 >= 247 && b0 <= 250) {
        auto v = ((i64)b0 - 247) << 8 + src[1] + 108;
        src += 2;
        return v;
    }
    if(b0 >= 251 && b0 <= 254) {

        auto v = -((i64)b0 - 251) << 8 - src[1] - 108;
        src += 2;
        return v;
    }
    if(b0 == 28) {

        auto v = (i64)src[1] << 8 | (i64)src[2];
        src += 3;
        return v;
    }
    if(b0 == 29) {
        auto v = (i64)src[1] << 24 | (i64)src[2] << 16 | (i64)src[3] << 8 | (i64)src[4];
        src += 5;
        return v;
    }
    ASSERT(0);
}
i64 GetIntOperandCFF(DictionaryCFF* dict) {

    u8 b0 = dict->it[0];
    if(b0 >= 32 && b0 <= 246) {
        dict->it += 1;
        return (i64)b0 - 139;
    }
    if(b0 >= 247 && b0 <= 250) {
        auto v = ((i64)b0 - 247) * 256 + dict->it[1] + 108;
        dict->it += 2;
        return v;
    }
    if(b0 >= 251 && b0 <= 254) {

        i64 v = ((i64)b0 - 251) * (-256) - dict->it[1] - 108;
        dict->it += 2;
        return v;
    }
    if(b0 == 28) {

        i64 v = (i64)dict->it[1] << 8 | (i64)dict->it[2];
        dict->it += 3;
        return v;
    }
    if(b0 == 29) {

        i64 v = (i64)dict->it[1] << 24 | (i64)dict->it[2] << 16 | (i64)dict->it[3] << 8 | (i64)dict->it[4];
        dict->it += 5;
        return v;
    }
    ASSERT(0);
}
u32 GetOperandSize(u8* src) {

    u8 b0 = src[0];
    if(b0 == 30) {

        auto it = src;
        while( ((*it & 0xF) != 0xF) && (*it != 0xFF) ) {
            it++;
        }
        return (it - src) + 1;
    }
    else {

        if(b0 >= 32 && b0 <= 246) {
            return 1;
        }
        if(b0 >= 247 && b0 <= 250) {
            return 2;
        }
        if(b0 >= 251 && b0 <= 254) {
            return 2;
        }
        if(b0 == 28) {
            return 3;
        }
        if(b0 == 29) {
            return 5;
        }
    }

    return ~u32(0);
}
f64 GetRealOperandCFF(byte* entry) {

}
void GetIntArrayDictCFF(DictionaryCFF dict, u32 count, i64* dst) {

    for(u32 i = 0; i < count; i++) {
        dst[i] = GetIntOperandCFF(&dict);
    }
}
byte* GetCFFDictEntry(DictionaryCFF dict, u16 key) {

    auto it = dict.begin;
    u32 i = 0;
    for(;it <= dict.end; i++) {

        auto begin = it;
        while(*it >= 28) {
            it += GetOperandSize(it);
        }

        u16 dictOp = *(it++);
        if(dictOp == 12) {
            dictOp = *(it++) | 0x100;
        }

        if(dictOp == key) {
            return begin;
        }
    }

    return nullptr;
}
CFFInfo ParseCFF(HeaderCFF* cff) {

    CFFInfo ret{};
    ret.head.offSize = cff->offSize;
    ret.head.hdrSize = cff->hdrSize;

    ret.nameIndex             = (IndexCFF*)((byte*)cff + ret.head.hdrSize);
    ret.topDictIndex          = (IndexCFF*)GetEndofCFFIndex(ret.nameIndex);
    ret.stringIndex           = (IndexCFF*)GetEndofCFFIndex(ret.topDictIndex);
    ret.globalSubroutineIndex = (IndexCFF*)GetEndofCFFIndex(ret.stringIndex);

    auto topDict = GetIndexEntryCFF(ret.topDictIndex, 0);
    DictionaryCFF dict{
        .begin = topDict.begin,
        .end = topDict.end,
        .it = topDict.begin
    };

    i64 charStrings     = 0;
    i64 charStringsType = 2;
    i64 fdarrayoff      = 0;
    i64 fdselectoff     = 0;

    auto val0 = GetIntCFF(GetCFFDictEntry(dict, 17));
    auto val1 = GetIntCFF(GetCFFDictEntry(dict, 0x100 | 6));
    auto val2 = GetIntCFF(GetCFFDictEntry(dict, 0x100 | 36));
    auto val3 = GetIntCFF(GetCFFDictEntry(dict, 0x100 | 37));

    charStrings     = (val0 == (i64(1) << 63)) ? charStrings     : val0;
    charStringsType = (val1 == (i64(1) << 63)) ? charStringsType : val1;
    fdarrayoff      = (val2 == (i64(1) << 63)) ? fdarrayoff      : val2;
    fdselectoff     = (val3 == (i64(1) << 63)) ? fdselectoff     : val3;

    if(charStringsType != 2 || charStrings == 0) return {};
    ret.charStringIndex = (IndexCFF*)((byte*)cff + charStrings);

    dict.it = GetCFFDictEntry(dict, 18);
    i64 sizeOffset[2] = {};
    GetIntArrayDictCFF(dict, 2, sizeOffset);

    ret.privateDict = {
        .begin = (byte*)cff + sizeOffset[1],
        .end   = (byte*)cff + sizeOffset[0] + sizeOffset[1],
        .it    = (byte*)cff + sizeOffset[1],
    };

    ret.subroutineIndex = (IndexCFF*)(ret.privateDict.begin + GetIntCFF(GetCFFDictEntry(ret.privateDict, 19)));
    if(fdarrayoff) {
        ret.fontDictIndex = (IndexCFF*)((byte*)cff + fdarrayoff);
    }

    return ret;
}
HeadTableOTF LoadHeadTableOTF(byte* mem) {

    auto src = (HeadTableOTF*)mem;
    HeadTableOTF ret{};

    ret.majorVersion        = ReverseByteOrder(src->majorVersion);
    ret.minorVersion        = ReverseByteOrder(src->minorVersion);
    ret.fixedFontRevision   = ReverseByteOrder(src->fixedFontRevision);
    ret.checksumAdjustment  = ReverseByteOrder(src->checksumAdjustment);

    ret.magicNum            = ReverseByteOrder(src->magicNum); // 0x5F0F3CF5
    ret.flags               = ReverseByteOrder(src->flags);
    ret.unitsPerEm          = ReverseByteOrder(src->unitsPerEm);

    ret.created             = ReverseByteOrder(src->created);
    ret.modified            = ReverseByteOrder(src->modified);

    ret.xMin                = ReverseByteOrder(src->xMin);
    ret.yMin                = ReverseByteOrder(src->yMin);
    ret.xMax                = ReverseByteOrder(src->xMax);
    ret.yMax                = ReverseByteOrder(src->yMax);

    ret.macStyle            = ReverseByteOrder(src->macStyle);
    ret.lowestRecPPEM       = ReverseByteOrder(src->lowestRecPPEM);
    ret.fontDirectionHint   = ReverseByteOrder(src->fontDirectionHint);
    ret.locaFormat          = ReverseByteOrder(src->locaFormat);
    ret.glyphDataFormat     = ReverseByteOrder(src->glyphDataFormat);

    return ret;
}


OTFInfo ParseOTFMemory(byte* mem, u32 size, LinearAllocator* alloc) {

    OTFInfo ret{};
    auto dirs = (TableDirectoryOTF*)mem;
    ret.begin = mem;
    ret.tableRecords = dirs->tableRecords;

    ret.sfntVersion   = ReverseByteOrder(dirs->sfntVersion);
    ret.tableCount    = ReverseByteOrder(dirs->tableCount);
    ret.searchRange   = ReverseByteOrder(dirs->searchRange);
    ret.entrySelector = ReverseByteOrder(dirs->entrySelector);
    ret.rangeShift    = ReverseByteOrder(dirs->rangeShift);


    for(u32 k = 0; k < SIZE_OF_ARRAY(OTF_TABLES); k++) {
        auto table = GetTableOTF(&ret, OTF_TABLES[k], dirs);
        ret.tables[k] = table;
        if(table) {
            auto checkSum = ReverseByteOrder(table->checkSum);
            auto computedCheckSum = ComputeTableChecksumOTF((u32*)table, ReverseByteOrder(table->len));
            if(computedCheckSum != checkSum) {

            }
        }

    }

    ASSERT(ret.head);
    ret.headTable = LoadHeadTableOTF(mem + ReverseByteOrder(ret.head->offset));

    auto offset = ReverseByteOrder(ret.cmap->offset);
    auto cmap = (cmapTableOTF*)(mem + offset);
    auto tableCount = ReverseByteOrder(cmap->numTables);

    for(u32 i = 0; i < tableCount; i++) {

        auto subTableOffset = ReverseByteOrder(cmap->encodings[i].subTableOffset);
        auto subTable = (byte*)cmap + subTableOffset;
        auto platformID = ReverseByteOrder(cmap->encodings[i].platformID);
        auto encodingID = ReverseByteOrder(cmap->encodings[i].encodingID);
        switch(platformID) {
        case OTF_PLATFORM_UNICODE:
            ParseUnicodePlatformSubTableOTF(&ret, subTable, encodingID, alloc);
            i = tableCount; // break from loop
            break;
        case OTF_PLATFORM_MAC:
        case OTF_PLATFORM_ISO:
        case OTF_PLATFORM_WINDOWS:
        case OTF_PLATFORM_CUSTOM:
            break;
        }
    }

    if(!ret.glyf) {
        ASSERT(ret.cff);

        auto header = (HeaderCFF*)(mem + ReverseByteOrder(ret.cff->offset));
        ret.cffInfo = (CFFInfo*)linear_allocate(alloc, sizeof(CFFInfo));
        *ret.cffInfo = ParseCFF(header);
    }

    return ret;
}


struct CharState {

    f32 firstX;
    f32 firstY;
    f32 x;
    f32 y;
};
struct CharStringMachine {

    u32 clear;
    f32 array[32];
    f32 stackSlots[48];
    CFFIndexEntry addressStack[10];
    u32 addressTop;
    u32 top;
    u32 vertexCount;

    CharState charState;
};
bool IsByteOperator(byte b) {

    return  b >= 0 && b <= 11   |
            b >= 13 && b <= 18  |
            b >= 21 && b <= 27  |
            b >= 29 && b <= 31;
}
enum CharStringOperators : u8 {

    // 1 byte ops
    OP_HSTEM = 1,
    OP_VSTEM = 3,
    OP_VMOVETO,
    OP_RLINETO,
    OP_HLINETO,
    OP_VLINETO,
    OP_RRCURVETO,
    OP_CALLSUBR = 10,
    OP_RETURN,
    OP_ESCAPE,
    OP_ENDCHAR = 14,
    OP_HSTEMHM = 18,
    OP_HINTMASK,
    OP_CNTRMASK,
    OP_RMOVETO,
    OP_HMOVETO,
    OP_VSTEMHM,
    OP_RCURVELINE,
    OP_RLINECURVE,
    OP_VVCURVETO,
    OP_HHCURVETO,
    OP_CALLGSUBR = 29,
    OP_VHCURVETO,
    OP_HVCURVETO,
    // 2 byte ops
    OP_AND = 3,
    OP_OR,
    OP_NOT,
    OP_ABS = 9,
    OP_ADD,
    OP_SUB,
    OP_DIV,
    OP_NEG = 14,
    OP_EQ,
    OP_DROP = 18,
    OP_PUT = 20,
    OP_GET,
    OP_IFELSE,
    OP_RANDOM,
    OP_MUL,
    OP_SQRT = 26,
    OP_DUP,
    OP_EXCH,
    OP_INDEX,
    OP_ROLL,
    OP_HLFEX = 34,
    OP_FLEX,
    OP_HFLEX1,
    OP_FLEX1,
};
void NewVertexPoint(CharStringMachine* machine, LinearAllocator* alloc, u32 type, f32 dx, f32 dy) {

    machine->vertexCount++;

    auto v = (GylphVertexPoint*)linear_allocate(alloc, sizeof(GylphVertexPoint));
    v->type = type;
    v->p.x = machine->charState.x + dx;
    v->p.y = machine->charState.y + dy;
    machine->charState.x += dx;
    machine->charState.y += dy;

}
void NewVertexQuadBezier(CharStringMachine* machine, LinearAllocator* alloc, f32 dx0, f32 dy0, f32 dx1, f32 dy1) {

    machine->vertexCount++;
    auto v = (GylphVertexQuadraticBezier*)linear_allocate(alloc, sizeof(GylphVertexQuadraticBezier));
    v->type = GLYPH_VERTEX_QUAD_BEZIER;
    v->c0.x = machine->charState.x + dx0;
    v->c0.y = machine->charState.y + dy0;
    v->c1.x = v->c0.x + dx1;
    v->c1.y = v->c0.y + dy1;

    machine->charState.x = v->c1.x;
    machine->charState.y = v->c1.y;

}
void NewVertexCubicBezier(CharStringMachine* machine, LinearAllocator* alloc, f32 dx0, f32 dy0, f32 dx1, f32 dy1, f32 dx2, f32 dy2) {

    machine->vertexCount++;
    auto v = (GylphVertexCubicBezier*)linear_allocate(alloc, sizeof(GylphVertexCubicBezier));

    v->type = GLYPH_VERTEX_CUBIC_BEZIER;
    v->c0.x = machine->charState.x + dx0;
    v->c0.y = machine->charState.y + dy0;
    v->c1.x = v->c0.x + dx1;
    v->c1.y = v->c0.y + dy1;
    v->c2.x = v->c1.x + dx2;
    v->c2.y = v->c1.y + dy2;

    machine->charState.x = v->c2.x;
    machine->charState.y = v->c2.y;
}

v2 CalcBezier3(v2* curve, f32 t) {

    f32 p0x = curve[0].x;
    f32 p0y = curve[0].y;

    f32 p1x = curve[1].x;
    f32 p1y = curve[1].y;

    f32 p2x = curve[2].x;
    f32 p2y = curve[2].y;

    f32 t_ = 1.f - t;
    f32 x = t_ * (t_ * p0x + t * p1x) + t * (t_ * p1x + t * p2x);
    f32 y = t_ * (t_ * p0y + t * p1y) + t * (t_ * p1y + t * p2y);

    return {x,y};
}
v2 CalcBezier4(v2* curve, f32 t) {

    f32 p0x = curve[0].x;
    f32 p0y = curve[0].y;

    f32 p1x = curve[1].x;
    f32 p1y = curve[1].y;

    f32 p2x = curve[2].x;
    f32 p2y = curve[2].y;

    f32 p3x = curve[3].x;
    f32 p3y = curve[3].y;

    f32 t_ = (1.f - t);

    f32 b0x = t_ * (t_ * p0x + t * p1x) + t * (t_ * p1x + t * p2x);
    f32 b0y = t_ * (t_ * p0y + t * p1y) + t * (t_ * p1y + t * p2y);

    f32 b1x = t_ * (t_ * p1x + t * p2x) + t * (t_ * p2x + t * p3x);
    f32 b1y = t_ * (t_ * p1y + t * p2y) + t * (t_ * p2y + t * p3y);

    return {t_ * b0x + t * b1x, t_ * b0y + t * b1y};
}
u32 TesselateQuadBezier(LinearAllocator* alloc, f32 x0, f32 y0, f32 x1, f32 y1, f32 x2, f32 y2, f32 flatnessSquared, u32 n) {

    // midpoint
    float mx = (x0 + 2*x1 + x2)/4;
    float my = (y0 + 2*y1 + y2)/4;
    // versus directly drawn line

    float dx = (x0+x2)/2 - mx;
    float dy = (y0+y2)/2 - my;
    if (n > 16) {
        // 65536 segments on one curve better be enough!
        return 0;
    }
    if (dx*dx+dy*dy > flatnessSquared) { // half-pixel error allowed... need to be smaller if AA
        u32 ret = 0;
        ret += TesselateQuadBezier(alloc, x0,y0, (x0+x1)/2.0f,(y0+y1)/2.0f, mx,my, flatnessSquared, n+1);
        ret += TesselateQuadBezier(alloc, mx,my, (x1+x2)/2.0f,(y1+y2)/2.0f, x2,y2, flatnessSquared, n+1);
        return ret;
    }
    else {
        auto p = (v2*)linear_allocate(alloc, sizeof(v2));
        p->x = x2;
        p->y = y2;
        return 1;
    }
}
u32 TesslateCubicBezier(LinearAllocator* alloc, f32 x0, f32 y0, f32 x1, f32 y1, f32 x2, f32 y2, f32 x3, f32 y3, f32 objspace_flatness_squared, int n) {

    f32 dx0 = x1-x0;
    f32 dy0 = y1-y0;
    f32 dx1 = x2-x1;
    f32 dy1 = y2-y1;
    f32 dx2 = x3-x2;
    f32 dy2 = y3-y2;
    f32 dx = x3-x0;
    f32 dy = y3-y0;
    f32 longlen = (f32) (sqrt(dx0*dx0+dy0*dy0)+sqrt(dx1*dx1+dy1*dy1)+sqrt(dx2*dx2+dy2*dy2));
    f32 shortlen = (f32) sqrt(dx*dx+dy*dy);
    f32 flatness_squared = longlen*longlen-shortlen*shortlen;

    if (n > 16) {
        // 65536 segments on one curve better be enough!
        return 0;
    }

    if (flatness_squared > objspace_flatness_squared) {
        f32 x01 = (x0+x1)/2;
        f32 y01 = (y0+y1)/2;
        f32 x12 = (x1+x2)/2;
        f32 y12 = (y1+y2)/2;
        f32 x23 = (x2+x3)/2;
        f32 y23 = (y2+y3)/2;

        f32 xa = (x01+x12)/2;
        f32 ya = (y01+y12)/2;
        f32 xb = (x12+x23)/2;
        f32 yb = (y12+y23)/2;

        f32 mx = (xa+xb)/2;
        f32 my = (ya+yb)/2;
        u32 ret = 0;
        ret += TesslateCubicBezier(alloc, x0,y0, x01,y01, xa,ya, mx,my, objspace_flatness_squared, n+1);
        ret += TesslateCubicBezier(alloc, mx,my, xb,yb, x23,y23, x3,y3, objspace_flatness_squared, n+1);
        return ret;
    }
    else {
        auto p = (v2*)linear_allocate(alloc, sizeof(v2));
        p->x = x3;
        p->y = y3;
        return 1;
    }
}
void TesselateQuadBezierEqualSeg(u32 resolution, v2* curve, LinearAllocator* alloc) {

    f32 stepT = 1.f / (f32)resolution;
    f32 t = 0;
    for(u32 i = 0; i < resolution; i++) {

        auto p = CalcBezier3(curve, t);
        auto m = (v2*)linear_allocate(alloc, sizeof(v2));
        *m = p;

        t += stepT;
    }
}
void TesselateCubicBezierEqualSeg(u32 resolution, v2* curve, LinearAllocator* alloc) {

    f32 stepT = 1.f / (f32)resolution;
    f32 t = 0;
    for(u32 i = 0; i < resolution; i++) {

        auto p = CalcBezier4(curve, t);
        auto m = (v2*)linear_allocate(alloc, sizeof(v2));
        *m = p;

        t += stepT;
    }
}

u32 TesselateOutline(GlyphOutline outline, u32 contourPointCount[], f32 flattnessThreshold, LinearAllocator* alloc) {

    u32* it = (u32*)(outline.begin);
    f32 x = 0;
    f32 y = 0;
    f32 firstX = 0;
    f32 firstY = 0;
    u32 top = alloc->top;
    f32 flattnessSqr = flattnessThreshold * flattnessThreshold;

    i32 contourIndex = -1;
    memset(contourPointCount, 0, outline.contourCount * sizeof(u32));
    u32 ret = 0;
    u32 start = 0;

    for(u32 i = 0; i < outline.vertexCount && (byte*)it <= outline.end; i++) {

        switch(*it) {
        case GLYPH_MOVE_TO:
            {
                contourPointCount[++contourIndex] = 1;
                ret++;
                if(contourIndex > 0) {
                    ret--;
                    contourPointCount[contourIndex - 1]--;
                    alloc->top -= sizeof(v2);
                }

                auto v = (GylphVertexPoint*)it;
                x = v->p.x;
                y = v->p.y;
                auto point = (v2*)linear_allocate(alloc, sizeof(v2));
                point->x = x;
                point->y = y;
                firstX = x;
                firstY = y;
                it += sizeof(GylphVertexPoint) / sizeof(u32);

                break;
            }
        case GLYPH_VERTEX_POINT:
            {
                contourPointCount[contourIndex]++;
                ret++;

                auto v = (GylphVertexPoint*)it;
                x = v->p.x;
                y = v->p.y;
                auto point = (v2*)linear_allocate(alloc, sizeof(v2));
                point->x = x;
                point->y = y;

                it += sizeof(GylphVertexPoint) / sizeof(u32);
                break;
            }
        case GLYPH_VERTEX_QUAD_BEZIER:
            {
                auto v = (GylphVertexQuadraticBezier*)it;
                u32 pCount = TesselateQuadBezier(alloc, x, y, v->c0.x, v->c0.y, v->c1.x, v->c1.y, flattnessSqr, 0);
                contourPointCount[contourIndex] += pCount;
                ret += pCount;

                x = v->c1.x;
                y = v->c1.y;

                it += sizeof(GylphVertexQuadraticBezier) / sizeof(u32);
                break;
            }
        case GLYPH_VERTEX_CUBIC_BEZIER:
            {
                auto v = (GylphVertexCubicBezier*)it;
                u32 pCount = TesslateCubicBezier(alloc, x,y, v->c0.x, v->c0.y, v->c1.x, v->c1.y, v->c2.x, v->c2.y, flattnessSqr, 0);
                contourPointCount[contourIndex] += pCount;
                ret += pCount;

                x = v->c2.x;
                y = v->c2.y;
                it += sizeof(GylphVertexCubicBezier) / sizeof(u32);
                break;
            }
        }
    }

    contourPointCount[contourIndex]--;
    alloc->top -= sizeof(v2);
    return ret - 1;
}

i64 QPartitionPointsY(v2* arr, i64 low, i64 high) {

    f32 pivot = arr[high].y;
    i64 index = low - 1;

    for(i64 i = low; i < high; i++) {
        if(arr[i].y < pivot) {
            index++;
            auto tmp = arr[index];
            arr[index] = arr[i];
            arr[i] = tmp;
        }
    }

    auto tmp = arr[high];
    arr[high] = arr[index + 1];
    arr[index + 1] = tmp;
    return index + 1;
}
void QsortPointsY(v2* arr, i64 low, i64 high) {

    if(low < high) {
        i64 pivot = QPartitionPointsY(arr, low, high);
        QsortPointsY(arr, low, pivot-1);
        QsortPointsY(arr, pivot+1, high);
    }
}
i64 QPartitionPointsX(v2* arr, i64 low, i64 high) {

    f32 pivot = arr[high].x;
    i64 index = low - 1;

    for(i64 i = low; i < high; i++) {
        if(arr[i].x < pivot) {
            index++;
            auto tmp = arr[index];
            arr[index] = arr[i];
            arr[i] = tmp;
        }
    }

    auto tmp = arr[high];
    arr[high] = arr[index + 1];
    arr[index + 1] = tmp;
    return index + 1;
}
void QsortPointsX(v2* arr, i64 low, i64 high) {

    if(low < high) {
        i64 pivot = QPartitionPointsX(arr, low, high);
        QsortPointsX(arr, low, pivot-1);
        QsortPointsX(arr, pivot+1, high);
    }
}

u32 MakeEdgelistFromOutline(GlyphOutline outline, f32 flattnessThreshold, u32 contours[], Edge** res, LinearAllocator* alloc) {

    u32 top = alloc->top;
    u32 pCount = TesselateOutline(outline, contours, flattnessThreshold, alloc);

    // Edge* edgeDst = (Edge*)linear_allocator_top(alloc);
    // auto edgeCount = Unwind((v2*)(alloc->base + top), contours, outline.contourCount, {0,0}, 0, alloc);
    // *res = edgeDst;

    return pCount;
}

void ClosePath(CharStringMachine* machine, LinearAllocator* alloc) {

    auto firstX = machine->charState.firstX;
    auto firstY = machine->charState.firstY;
    auto x = machine->charState.x;
    auto y = machine->charState.y;

    if(firstX != x || firstY != y) {

        machine->vertexCount++;
        auto v = (GylphVertexPoint*)linear_allocate(alloc, sizeof(GylphVertexPoint));
        v->type = GLYPH_VERTEX_POINT;
        v->p.x = firstX;
        v->p.y = firstY;
    }
}
union f32_u32_union {
    f32 f;
    u32 u;
};
CFFIndexEntry LookupSubrCFF(IndexCFF* index, i32 subr) {

    auto count = ReverseByteOrder(index->count);
    i32 bias = 107;
    if(count >= 3390) {
        bias = 32768;
    }
    else if(count >= 1240) {
        bias = 1131;
    }
    subr += bias;
    ASSERT(subr && subr <= count);
    return GetIndexEntryCFF(index, subr);
}

GlyphOutline ExecuteCharStringProgram(CharStringMachine* machine, CFFIndexEntry program, IndexCFF* globalSubrs, IndexCFF* subrs, LinearAllocator* alloc) {

    GlyphOutline ret;
    ret.contourCount = 0;
    ret.begin = (byte*)linear_allocator_top(alloc);
    const auto ogP = program;
    u32 instCounter = 0;

    auto it = program.begin;
    while(it < program.end) {

        machine->clear = 0;
        switch(*it) {
        // TODO() implement hints
        // hints
        case OP_HSTEM:
        case OP_VSTEM:
        case OP_HSTEMHM:
        case OP_VSTEMHM:
        case OP_HINTMASK:
        case OP_CNTRMASK:
            it++;
            break;

        case OP_CALLSUBR:
        case OP_CALLGSUBR:
            {
                machine->clear = ~u32(0);
                IndexCFF* subrIndex = (*it == OP_CALLSUBR) ? subrs : globalSubrs;
                it++;
                i64 subr = machine->stackSlots[--machine->top];
                auto subrProgram = LookupSubrCFF(subrIndex, subr);

                LOG_ASSERT(machine->addressTop < 10, "char string program hit recursion limit");
                machine->addressStack[machine->addressTop++] = {it, program.end};

                it = subrProgram.begin;
                program.end = subrProgram.end;
                break;
            }
        case OP_ENDCHAR:
            it++;
            ClosePath(machine, alloc);
            break;

        case OP_RETURN:
            LOG_ASSERT(machine->addressTop > 0, "char string program subroutine return address stack underflow");
            it = machine->addressStack[--machine->addressTop].begin;
            program.end = machine->addressStack[machine->addressTop].end;
            machine->clear = ~u32(0);
            break;
        case OP_ESCAPE:
            it++;
            // TODO() implement 2 byte ops
            switch(*it) {
            case OP_AND:
            case OP_OR:
            case OP_NOT:
            case OP_ABS:
            case OP_ADD:
            case OP_SUB:
            case OP_DIV:
            case OP_NEG:
            case OP_EQ:
            case OP_DROP:
            case OP_PUT:
            case OP_GET:
            case OP_IFELSE:
            case OP_RANDOM:
            case OP_MUL:
            case OP_SQRT:
            case OP_DUP:
            case OP_EXCH:
            case OP_INDEX:
            case OP_ROLL:

            case OP_HLFEX:
            case OP_FLEX:
            case OP_HFLEX1:
            case OP_FLEX1:
                ASSERT(false);
                break;
            }
            break;

        case OP_RMOVETO:
            it++;
            ClosePath(machine, alloc);
            machine->charState.x += machine->stackSlots[machine->top-2];
            machine->charState.y += machine->stackSlots[machine->top-1];
            machine->charState.firstX = machine->charState.x;
            machine->charState.firstY = machine->charState.y;
            ret.contourCount++;
            NewVertexPoint(machine, alloc, GLYPH_MOVE_TO, 0,0);
            break;
        case OP_VMOVETO:
            it++;
            ClosePath(machine, alloc);
            machine->charState.y += machine->stackSlots[machine->top-1];
            machine->charState.firstX = machine->charState.x;
            machine->charState.firstY = machine->charState.y;
            ret.contourCount++;
            NewVertexPoint(machine, alloc, GLYPH_MOVE_TO, 0,0);
            break;
        case OP_HMOVETO:
            it++;
            ClosePath(machine, alloc);
            machine->charState.x += machine->stackSlots[machine->top-1];
            machine->charState.firstX = machine->charState.x;
            machine->charState.firstY = machine->charState.y;
            ret.contourCount++;
            NewVertexPoint(machine, alloc, GLYPH_MOVE_TO, 0,0);
            break;

        case OP_RLINETO:
            it++;
            for(u32 i = 0; i < machine->top; i += 2) {
                auto dx = machine->stackSlots[i + 0];
                auto dy = machine->stackSlots[i + 1];

                NewVertexPoint(machine, alloc, GLYPH_VERTEX_POINT, dx, dy);
            }
            break;
        case OP_HLINETO:
        case OP_VLINETO:
            {
                u32 mask = 0 - (*it++ == OP_HLINETO);
                if(machine->top & 1 == 1) {
                    auto d0 = Mem<f32_u32_union>(machine->stackSlots);
                    auto d1 = d0;
                    d0.u &= mask;
                    d1.u &= ~mask;

                    NewVertexPoint(machine, alloc, GLYPH_VERTEX_POINT, d0.f, d1.f);
                }
                for(u32 i = (machine->top & 1); i < machine->top; i++) {

                    auto d0 = Mem<f32_u32_union>(machine->stackSlots + i);
                    auto d1 = d0;

                    d0.u &= mask;
                    mask = ~mask;
                    d1.u &= mask;

                    NewVertexPoint(machine, alloc, GLYPH_VERTEX_POINT, d0.f, d1.f);
                }
                break;
            }
        case OP_RRCURVETO:
            it++;
            for(u32 i = 0; i < machine->top; i += 6) {
                NewVertexCubicBezier(machine, alloc,
                    machine->stackSlots[i+0], machine->stackSlots[i+1],
                    machine->stackSlots[i+2], machine->stackSlots[i+3],
                    machine->stackSlots[i+4], machine->stackSlots[i+5]
                );
            }
            break;
        case OP_VVCURVETO:
            {
                it++;
                f32 dx0 = 0;
                if(machine->top & 1 == 1) {
                    dx0 = machine->stackSlots[0];
                }
                for(u32 i = (machine->top & 1); i < machine->top; i += 4) {
                    f32 dy0 = machine->stackSlots[i];
                    f32 dx1 = machine->stackSlots[i + 1];
                    f32 dy1 = machine->stackSlots[i + 2];
                    f32 dy2 = machine->stackSlots[i + 3];

                    NewVertexCubicBezier(machine, alloc,
                        dx0, dy0,
                        dx1, dy1,
                        0, dy2
                    );
                    dx0 = 0;
                }
            }
            break;
        case OP_HHCURVETO:
            {
                it++;
                f32 dy0 = 0;
                if(machine->top & 1 == 1) {
                    dy0 = machine->stackSlots[0];
                }
                for(u32 i = (machine->top & 1); i < machine->top; i += 4) {
                    f32 dx0 = machine->stackSlots[i];
                    f32 dx1 = machine->stackSlots[i + 1];
                    f32 dy1 = machine->stackSlots[i + 2];
                    f32 dx2 = machine->stackSlots[i + 3];

                    NewVertexCubicBezier(machine, alloc,
                        dx0, dy0,
                        dx1, dy1,
                        dx2, 0
                    );
                    dy0 = 0;
                }
            }
            break;
        case OP_VHCURVETO:
        case OP_HVCURVETO:
            {
                auto args = machine->top - ((machine->top & 1) ? 5 : 0);
                u32 HV = (*it++ == OP_HVCURVETO);
                u32 mask = 0 - HV;
                u32 i = 0;
                for(; i < args; i += 4) {

                    auto da0 = Mem<f32_u32_union>(machine->stackSlots + i);
                    auto da1 = da0;
                    da0.u &= mask;
                    da1.u &= ~mask;

                    f32 dxb = machine->stackSlots[i + 1];
                    f32 dyb = machine->stackSlots[i + 2];

                    auto dc0 = Mem<f32_u32_union>(machine->stackSlots + i + 3);
                    auto dc1 = dc0;
                    dc0.u &= ~mask;
                    dc1.u &= mask;

                    NewVertexCubicBezier(machine, alloc,
                        da0.f, da1.f,
                        dxb, dyb,
                        dc0.f, dc1.f
                    );

                    mask = ~mask;
                }
                if(machine->top & 1) {
                    auto da0 = Mem<f32_u32_union>(machine->stackSlots + i);
                    auto da1 = da0;
                    da0.u &= mask;
                    da1.u &= ~mask;

                    f32 dxb = machine->stackSlots[i + 1];
                    f32 dyb = machine->stackSlots[i + 2];

                    // vh
                    // o  e  o
                    // xy yx xy
                    // vh hv vh
                    // 34 43 34

                    // hv
                    // o  e  o
                    // yx xy yx
                    // hv vh hv
                    // 43 34 43

                    u32 odd = !((i >> 2) & 1);
                    u32 coordOff = (HV == odd);

                    auto dc0 = machine->stackSlots[i + 3 +  coordOff]; // 0
                    auto dc1 = machine->stackSlots[i + 3 + !coordOff]; // 1

                    NewVertexCubicBezier(machine, alloc,
                        da0.f, da1.f,
                        dxb, dyb,
                        dc0, dc1
                    );
                }
            }
            break;

        case OP_RCURVELINE:
            {

                it++;
                u32 curvePointCount = machine->top - 2;
                u32 i = 0;
                for(; i < curvePointCount; i += 4) {
                    NewVertexCubicBezier(machine, alloc,
                        machine->stackSlots[i+0], machine->stackSlots[i+1],
                        machine->stackSlots[i+2], machine->stackSlots[i+3],
                        machine->stackSlots[i+4], machine->stackSlots[i+5]
                    );
                }
                auto dx = machine->stackSlots[i + 0];
                auto dy = machine->stackSlots[i + 1];
                NewVertexPoint(machine, alloc, GLYPH_VERTEX_POINT, dx, dy);
            }
            break;
        case OP_RLINECURVE:
            {
                it++;
                u32 linePointCount = machine->top - 4;
                u32 i = 0;
                for(; i < linePointCount; i += 2) {
                    auto dx = machine->stackSlots[i + 0];
                    auto dy = machine->stackSlots[i + 1];
                    NewVertexPoint(machine, alloc, GLYPH_VERTEX_POINT, dx, dy);
                }
                NewVertexCubicBezier(machine, alloc,
                    machine->stackSlots[i+0], machine->stackSlots[i+1],
                    machine->stackSlots[i+2], machine->stackSlots[i+3],
                    machine->stackSlots[i+4], machine->stackSlots[i+5]
                );
            }
            break;

        default:
            {
                ASSERT(*it == 255 || *it == 28 || *it >= 32);

                f32 f32imm;
                if(*it == 255) {
                    f32imm = (f32)((i32)ReadBytes(4, it+1) / 0x10000);
                    it += 5;
                }
                else {
                    DictionaryCFF d{};
                    d.begin = it;
                    d.it = it;
                    d.end = program.end;
                    f32imm = (f32)((i16)GetIntOperandCFF(&d));
                    it = d.it;
                }
                machine->stackSlots[machine->top++] = f32imm;
                machine->clear = ~u32(0);
                instCounter--;
                break;
            }
        }

        instCounter++;
        machine->top &= machine->clear;
    }

    ret.vertexCount = machine->vertexCount;
    ret.end = (byte*)linear_allocator_top(alloc);

    return ret;
}
GlyphOutline ExtractGlyphOutlineOTF(OTFInfo* info, u32 codePoint, LinearAllocator* res) {

    u32 glyphIndex;
    if(info->sparse) {
        glyphIndex = SearchGlyphIndexSparseMap(&info->sparseMap, codePoint);
    }
    else {
        glyphIndex = SearchGlyphIndexDenseMap(&info->denseMap, codePoint);
    }

    if(info->glyf) {

        u32 glyphOffset;
        if(info->headTable.locaFormat) {
            auto locations = (u32*)(info->begin + ReverseByteOrder(info->loca->offset));

            glyphOffset = locations[glyphIndex * 2 + 0];
            locations[glyphIndex * 2 + 1];
        }
        else {

            auto locations = (u16*)(info->begin + ReverseByteOrder(info->loca->offset));
            glyphOffset = locations[glyphIndex * 2 + 0];
            locations[glyphIndex * 2 + 1];
        }
    }
    else {

        CharStringMachine m{
            .clear        = {},
            .array        = {},
            .stackSlots   = {},
            .addressStack = {},
            .addressTop   = {},
            .top          = {},
            .vertexCount  = {},
            .charState = {
                .firstX = 0.0f,
                .firstY = 0.0f,
                .x      = 0.0f,
                .y      = 0.0f,
            }
        };
        auto program = GetIndexEntryCFF(info->cffInfo->charStringIndex, glyphIndex);
        return ExecuteCharStringProgram(&m, program, info->cffInfo->globalSubroutineIndex, info->cffInfo->subroutineIndex, res);
    }
}
f32 GetScaleForPixelHeight(OTFInfo* info, f32 height) {

    ASSERT(info->hhea);
    auto off = ReverseByteOrder(info->hhea->offset);
    auto h0 = ReverseByteOrder(Mem<i16>((info->begin + off) + 4));
    auto h1 = ReverseByteOrder(Mem<i16>((info->begin + off) + 6));
    i32 fheight = h0 - h1;
    return (f32)height / fheight;
}

u32 Unwind(v2* points, u32* wcount, u32 windings, v2 offset, bool invert, LinearAllocator* alloc) {

    // vsubsample should divide 255 evenly; otherwise we won't reach full opacity
    // now we have to blow out the windings into explicit edge lists

    int vsubsample = 1;
    u32 pCount = 0;
    for (u32 i = 0; i < windings; ++i) {
        pCount += wcount[i];
    }

    v2 maxBounds{0.0f, 0.0f};
    v2 sum{0.0f,0.0f};
    for(u32 i = 0, off = 0; i < windings; ++i) {

        auto p = points + off;
        off += wcount[i];
        for(u32 k = 0; k < wcount[i]; k++) {
            maxBounds.x = Max(maxBounds.x, p[k].x);
            maxBounds.y = Max(maxBounds.y, p[k].y);
            sum.x = p[k].x;
            sum.y = p[k].y;
        }
    }
    v2 scale{
        1.0f / maxBounds.x,
        1.0f / maxBounds.y
    };
    f32 y_scale_inv = invert ? -scale.y : scale.y;
    Edge* edges = (Edge*)linear_allocate(alloc, sizeof(Edge) * (pCount+1));
    u32 n = 0;
    u32 m = 0;

    for(u32 i = 0; i < windings; ++i) {

        auto p = points + m;
        m += wcount[i];
        u32 j = wcount[i]-1;

        for(u32 k = 0; k < wcount[i];) {

            int a = k;
            int b = j;

            // skip the edge if horizontal
            if(p[j].y == p[k].y) {
                j = k;
                k++;
                continue;
            }

            // add edge from j to k to the list
            if(invert) {
                if(p[j].y > p[k].y) {
                    a = j;
                    b = k;
                }
            }
            else if(p[j].y < p[k].y) {
                a = j;
                b = k;
            }

            edges[n].p0.x =  p[a].x * scale.x + offset.x;
            edges[n].p0.y = (p[a].y * y_scale_inv + offset.y) * vsubsample;
            edges[n].p1.x =  p[b].x * scale.x + offset.x;
            edges[n].p1.y = (p[b].y * y_scale_inv + offset.y) * vsubsample;

            n++;
            j = k;
            k++;
        }
    }

    auto last = n++;
    edges[last].p0 = {
        (points[0].x * scale.x) + offset.x,
        (points[0].y * scale.y) + offset.y
    };
    edges[last].p1 = {
        (points[pCount-1].x * scale.x) + offset.x,
        (points[pCount-1].y * scale.y) + offset.y
    };

    return n;
}
v2 LineLineSegmentIntersection(v2 rayOrigin, v2 rayDir, v2 a, v2 b) {

    v2 i0 = rayOrigin - a;
    v2 i1 = b - a;
    v2 i2 = {-rayDir.y, rayDir.x};

    f32 d = dot(i1, i2);
    if(d == 0.f) return {};
    f32 t1 = dot(i0, i2) / d;

    if(t1 > 0.f && t1 < 1.0f) {
        return a + (b - a) * t1;
    }
    return {};
}
v2 RayLineSegmentIntersection(v2 rayOrigin, v2 rayDir, v2 a, v2 b) {

    v2 i0 = rayOrigin - a;
    v2 i1 = b - a;
    v2 i2 = {-rayDir.y, rayDir.x};

    f32 d = dot(i1, i2);
    if(d == 0.f) return {};

    f32 t0 = cross(i1, i0) / d;
    f32 t1 = dot(i0, i2) / d;

    if(t0 > 0.0f && t1 > 0.0f && t1 < 1.0f) {
        return rayOrigin + rayDir * t0;
    }
    return {};
}

u32 TriangulateTesselatedContours(i32 (*dst)[3], u32 contourCount, u32 contourPointCount[], u32 pCount, v2* src, LinearAllocator* alloc) {

    f64 (*cpy)[2] = (f64(*)[2])linear_allocator_top(alloc);
    for(u32 i = 0; i < pCount; i++) {
        cpy[i][0] = src[i].x;
        cpy[i][1] = src[i].y;
    }
    return triangulate_polygon(contourCount, (i32*)contourPointCount, cpy - 1, dst);
}


/*
struct QNode;
struct SegmentQ;
struct TrapezoidQ;

struct QNode {

    // Y-node, S-node, sink-node
    u8 nodetype;

    // u32 segnum;
    SelfRelativePointer<SegmentQ> segment;

    // key value
    v2 yVal;

    //u32 trapezoidNum;
    SelfRelativePointer<TrapezoidQ> trapezoid;

    // doubly linked DAG
    SelfRelativePointer<QNode> parent;
    // children
    SelfRelativePointer<QNode> left;
    SelfRelativePointer<QNode> right;
};
struct TrapezoidQ {

    // two adjoining segments
    SelfRelativePointer<SegmentQ> leftSegment;
    SelfRelativePointer<SegmentQ> rightSegment;

    // max/min y-values
    v2 high;
    v2 low;

    SelfRelativePointer<TrapezoidQ> upper0;
    SelfRelativePointer<TrapezoidQ> upper1;
    SelfRelativePointer<TrapezoidQ> lower0;
    SelfRelativePointer<TrapezoidQ> lower1;

    // pointer to corresponding in Q
    SelfRelativePointer<QNode> sink;

    SelfRelativePointer<TrapezoidQ> usave;
    // int usave;
    int uside;
    int state;
};
struct SegmentQ {
    // two endpoints
    v2 v0;
    v2 v1;

    // root nodes in Q
    SelfRelativePointer<QNode> root0;
    SelfRelativePointer<QNode> root1;

    // Next logical segment
    SelfRelativePointer<SegmentQ> prev;
    // Previous segment
    SelfRelativePointer<SegmentQ> next;

    // inserted in trapezoidation yet?
    bool is_inserted;
};

enum QnodeType : u8 {
    Q_Y_NODE,
    Q_X_NODE,
    Q_SINK_NODE,
};
enum TrapezoidState : u8 {
    TRAPEZOID_VALID = 1,
};

// tolerance value: Used for making
#define C_EPS 1.0e-7
#define FP_EQUAL(s, t) (fabs(s - t) <= C_EPS)

#define CROSS(v0, v1, v2) (((v1).x - (v0).x)*((v2).y - (v0).y) - \
			   ((v1).y - (v0).y)*((v2).x - (v0).x))

#define DOT(v0, v1) ((v0).x * (v1).x + (v0).y * (v1).y)
#define S_LEFT 1		// for merge-direction
#define S_RIGHT 2

v2 MaxPoint(v2 v0, v2 v1) {

    if (v0.y > v1.y + C_EPS) {
        return v0;
    }
    else if(FP_EQUAL(v0.y, v1.y)) {
        if (v0.x > v1.x + C_EPS) {
            return v0;
        }
        else {
            return v1;
        }
    }
    else {
        return v1;
    }
}
v2 MinPoint(v2 v0, v2 v1)  {

    if (v0.y < v1.y - C_EPS) {
        return v0;
    }
    else if (FP_EQUAL(v0.y, v1.y)) {
        if (v0.x < v1.x) {
            return v0;
        }
        else
            return v1;
        }
    else {
        return v1;
    }
}
bool GreaterThanPoint(v2 v0, v2 v1) {

    if (v0.y > v1.y + C_EPS) {
        return true;
    }
    else if (v0.y < v1.y - C_EPS) {
        return false;
    }
    else {
        return (v0.x > v1.x);
    }
}


bool EqualPoint(v2 v0, v2 v1)  {
    return FP_EQUAL(v0.y, v1.y) && FP_EQUAL(v0.x, v1.x);
}

bool GreaterThanEqualPoint(v2 v0, v2 v1)  {

    if (v0.y > v1.y + C_EPS) {
        return TRUE;
    }
    else if (v0.y < v1.y - C_EPS) {
        return FALSE;
    }
    else {
        return (v0.x >= v1.x);
    }
}

bool LessThanPoint(v2 v0, v2 v1) {

    if(v0.y < v1.y - C_EPS) {
        return TRUE;
    }
    else if (v0.y > v1.y + C_EPS) {
        return FALSE;
    }
    else {
        return (v0.x < v1.x);
    }

}
int math_logstar_n(int n) {
	register int i;
	double v;

	for (i = 0, v = (double) n; v >= 1; i++)
		v = log2(v);

	return (i - 1);
}
int math_N(int n, int h) {
    register int i;
    double v;

    for (i = 0, v = (int) n; i < h; i++)
		v = log2(v);

    return (int) ceil((double) 1.0*n/v);
}

void MergeTrapezoids(SegmentQ* seg, TrapezoidQ* tfirst, TrapezoidQ* tlast, i32 side) {

    TrapezoidQ* it = tfirst;
    TrapezoidQ* tnext;
    QNode* ptnext;
    u32 cond;

    while(it && GreaterThanEqualPoint(it->low, tlast->low)) {

        if(side == S_LEFT) {

            cond = ((*it->lower0) && (*it->lower0->rightSegment == seg)) || (*it->lower1 && *it->lower1->rightSegment == seg);
        }
        else {

            cond = ((*it->lower0) && (*it->lower0->leftSegment == seg)) || (*it->lower1 && *it->lower1->leftSegment == seg);
        }

        if(cond) {

            if((*it->leftSegment == *tnext->leftSegment) && (*it->rightSegment)) {

                ptnext = *(tnext->sink->parent);

                if(*ptnext->left == *tnext->sink) {

                    ptnext->left = *it->sink;
                }
                else {

                    ptnext->right = *it->sink;
                }

                it->lower0 = *tnext->lower0;
                if(*it->lower0) {

                    if(*it->lower0->upper0 == tnext) {

                        it->lower0->upper0 = it;
                    }
                    else if(*it->lower0->upper1 == tnext) {

                        it->lower0->upper1 = it;
                    }
                }

                it->lower1 = *tnext->lower1;
                if(*it->lower1) {

                    if(*it->lower1->upper0 == tnext) {

                        it->lower1->upper0 = it;
                    }
                    else if(*it->lower1->upper1 == tnext) {

                        it->lower1->upper1 = it;
                    }
                }

                it->low = tnext->low;
                tnext->state = TRAPEZOID_VALID;

            }
            else {

                it = tnext;
            }
        }
        else {

            it = tnext;
        }


    }// end-while

}

bool IsLeftOf(SegmentQ* seg, v2 v) {

    f64 area;
    // seg. going upwards
    if (GreaterThanPoint(seg->v1, seg->v0)) {

        if(FP_EQUAL(seg->v1.y, v.y)) {
            if (v.x < seg->v1.x) {
                area = 1.0;
            }
            else {
                area = -1.0;
            }
        }
        else if(FP_EQUAL(seg->v0.y, v.y)) {
            if (v.x < seg->v0.x)
                area = 1.0;
            else {
                area = -1.0;
            }
        }
        else {
            area = CROSS(seg->v0, seg->v1, v);
        }
    }
    else {
        // v0 > v1

        if (FP_EQUAL(seg->v1.y, v.y)) {
            if (v.x < seg->v1.x) {
                area = 1.0;
            }
            else {
                area = -1.0;
            }
        }
        else if (FP_EQUAL(seg->v0.y, v.y)) {
            if (v.x < seg->v0.x) {
                area = 1.0;
            }
            else {
                area = -1.0;
            }
        }
        else {
            area = CROSS(seg->v1, seg->v0, v);
        }
    }

    if(area > 0.0) {
        return true;
    }
    else {
        return false;
    }
}
TrapezoidQ* LocateTrapezoid(QNode* root, v2 v, v2 vo) {

    switch(root->nodetype) {
    case Q_SINK_NODE:
        return *root->trapezoid;
    case Q_Y_NODE:

        // above
        if (GreaterThanPoint(v, root->yVal)) {
            return LocateTrapezoid(*root->right, v, vo);
        }
        else if (EqualPoint(v, root->yVal)) {
            // the point is already
            // inserted.
            if (GreaterThanPoint(vo, root->yVal))  {
                // above
                return LocateTrapezoid(*root->right, v, vo);
            }
            else {
                // below
                return LocateTrapezoid(*root->left, v, vo);
            }
        }
        else {
            // below
            return LocateTrapezoid(*root->left, v, vo);
        }

    case Q_X_NODE:
        if (EqualPoint(v, root->segment->v0) || EqualPoint(v, root->segment->v1)) {

            // horizontal segment
            if (FP_EQUAL(v.y, vo.y))  {
                if (vo.x < v.x) {
                    // left
                    return LocateTrapezoid(*root->left, v, vo);
                }
                else {
                    // right
                    return LocateTrapezoid(*root->right, v, vo);
                }
            }
            else if (IsLeftOf(*root->segment, vo)) {
                // left
                return LocateTrapezoid(*root->left, v, vo);
            }
            else {
                // right
                return LocateTrapezoid(*root->right, v, vo);
            }
        }
        else if(IsLeftOf(*root->segment, v)) {
            // left
            return LocateTrapezoid(*root->left, v, vo);
        }
        else {
            // right
            return LocateTrapezoid(*root->right, v, vo);
        }

    default:
        LOG_ASSERT(0, "Qnode type error");
        break;
    }
}
void AddSegment(QNode* root, SegmentQ* seg, MemoryPool<sizeof(TrapezoidQ)>* trPool, MemoryPool<sizeof(QNode)>* qPool) {

    v2 tpt;
    bool swapped = false;
    TrapezoidQ* firstTrapezoid = nullptr;
    TrapezoidQ* lastTrapezoid = nullptr;
    int triBottom = 0;

    if(GreaterThanPoint(seg->v1, seg->v0)) {

        tpt = seg->v0;
        seg->v0 = seg->v1;
        seg->v1 = tpt;

        auto tmp = *seg->root0;
        seg->root0 = *seg->root1;
        seg->root1 = tmp;
        swapped = true;
    }

    // insert v0
    if((swapped && seg->next->is_inserted) || (!swapped && seg->prev->is_inserted)) {

        auto trapezoidUpper = LocateTrapezoid(*seg->root0, seg->v0, seg->v1);
        auto trapezoidLower = (TrapezoidQ*)pool_allocate(trPool);
        *trapezoidLower = *trapezoidUpper;
        trapezoidLower->state = TRAPEZOID_VALID;

        trapezoidUpper->low = seg->v0;
        trapezoidLower->high = seg->v0;

        trapezoidUpper->lower0 = trapezoidLower;
        trapezoidUpper->lower1 = nullptr;

        trapezoidLower->upper0 = trapezoidUpper;
        trapezoidLower->upper1 = nullptr;

        if(*trapezoidLower->lower0) {

            if(*trapezoidLower->lower0->upper0 == trapezoidUpper) {
                trapezoidLower->lower0->upper0 = trapezoidLower;
            }
            if(*trapezoidLower->lower0->upper1 == trapezoidUpper) {
                trapezoidLower->lower0->upper1 = trapezoidLower;
            }
        }
        if(*trapezoidLower->lower1) {

            if(*trapezoidLower->lower1->upper0 == trapezoidUpper) {
                trapezoidLower->lower1->upper0 = trapezoidLower;
            }
            if(*trapezoidLower->lower1->upper1 == trapezoidUpper) {
                trapezoidLower->lower1->upper1 = trapezoidLower;
            }
        }

        auto upperSink = (QNode*)pool_allocate(qPool);
        auto lowerSink = (QNode*)pool_allocate(qPool);

        *upperSink = {};
        *lowerSink = {};

        auto sk = *trapezoidUpper->sink;
        sk->nodetype = Q_Y_NODE;
        sk->yVal = seg->v0;
        sk->segment = seg;
        sk->left = lowerSink;
        sk->right = upperSink;

        upperSink->nodetype = Q_SINK_NODE;
        upperSink->trapezoid = trapezoidUpper;
        upperSink->parent = sk;

        lowerSink->nodetype = Q_SINK_NODE;
        lowerSink->trapezoid = trapezoidLower;
        lowerSink->parent = sk;

        trapezoidUpper->sink = upperSink;
        trapezoidLower->sink = lowerSink;
        firstTrapezoid = trapezoidLower;
    }
    else {
        firstTrapezoid = LocateTrapezoid(*seg->root0, seg->v0, seg->v1);
    }

    // insert v1
    if((swapped && seg->prev->is_inserted) || (!swapped && seg->next->is_inserted)) {

        auto trapezoidUpper = LocateTrapezoid(*seg->root1, seg->v1, seg->v0);
        auto trapezoidLower = (TrapezoidQ*)pool_allocate(trPool);
        *trapezoidLower = *trapezoidUpper;
        trapezoidLower->state = TRAPEZOID_VALID;

        trapezoidUpper->low = seg->v1;
        trapezoidLower->high = seg->v1;

        trapezoidUpper->lower0 = trapezoidLower;
        trapezoidUpper->lower1 = nullptr;

        trapezoidLower->upper0 = trapezoidUpper;
        trapezoidLower->upper1 = nullptr;

        if(*trapezoidLower->lower0) {

            if(*trapezoidLower->lower0->upper0 == trapezoidUpper) {
                trapezoidLower->lower0->upper0 = trapezoidLower;
            }
            if(*trapezoidLower->lower0->upper1 == trapezoidUpper) {
                trapezoidLower->lower0->upper1 = trapezoidLower;
            }
        }
        if(*trapezoidLower->lower1) {

            if(*trapezoidLower->lower1->upper0 == trapezoidUpper) {
                trapezoidLower->lower1->upper0 = trapezoidLower;
            }
            if(*trapezoidLower->lower1->upper1 == trapezoidUpper) {
                trapezoidLower->lower1->upper1 = trapezoidLower;
            }
        }

        auto upperSink = (QNode*)pool_allocate(qPool);
        auto lowerSink = (QNode*)pool_allocate(qPool);

        *upperSink = {};
        *lowerSink = {};

        auto sk = *trapezoidUpper->sink;
        sk->nodetype = Q_Y_NODE;
        sk->yVal = seg->v1;
        sk->segment = seg;
        sk->left = lowerSink;
        sk->right = upperSink;

        upperSink->nodetype = Q_SINK_NODE;
        upperSink->trapezoid = trapezoidUpper;
        upperSink->parent = sk;

        lowerSink->nodetype = Q_SINK_NODE;
        lowerSink->trapezoid = trapezoidLower;
        lowerSink->parent = sk;

        trapezoidUpper->sink = upperSink;
        trapezoidLower->sink = lowerSink;
        lastTrapezoid = trapezoidLower;
    }
    else {
        firstTrapezoid = LocateTrapezoid(*seg->root0, seg->v0, seg->v1);
        triBottom = 1;
    }

    TrapezoidQ* firstRightTr = nullptr;
    TrapezoidQ* lastRightTr = nullptr;
    TrapezoidQ* firstLeftTr = nullptr;
    TrapezoidQ* lastLeftTr = nullptr;

    TrapezoidQ* tSav = nullptr;
    TrapezoidQ* tnSav = nullptr;
    SegmentQ* tmpTriSeg = nullptr;

    auto it = firstTrapezoid;
    while(it && GreaterThanEqualPoint(it->low, lastTrapezoid->low)) {

        TrapezoidQ* t_sav;
        TrapezoidQ* tn_sav;
        auto sk = *it->sink;
        auto leftQ = (QNode*)pool_allocate(qPool);
        auto rightQ = (QNode*)pool_allocate(qPool);

        *leftQ = {};
        *rightQ = {};
        sk->nodetype = Q_X_NODE;
        sk->segment = seg;
        sk->left = leftQ;
        sk->right = rightQ;

        leftQ->nodetype = Q_SINK_NODE;
        leftQ->trapezoid = it;
        leftQ->parent = sk;

        rightQ->nodetype = Q_SINK_NODE;
        auto rightTr = (TrapezoidQ*)pool_allocate(trPool); // tn
        *rightTr = {};
        rightTr->state = TRAPEZOID_VALID;
        rightQ->trapezoid = rightTr;
        rightQ->parent = sk;

        if(it == firstTrapezoid) {
            firstRightTr = rightTr;
        }
        if(EqualPoint(it->low, lastTrapezoid->low)) {
            lastRightTr = rightTr;
        }

        *rightTr = *it;
        it->sink = leftQ;
        rightTr->sink = rightQ;
        tSav = it;
        tnSav = rightTr;

        ASSERT(*it->lower0 || *it->lower1);
        if(*it->lower0 && *it->lower1 == nullptr) {

            if(*it->upper0 && *it->upper1) {

                if(*it->usave) {
                    if(it->uside == S_LEFT) {

                        rightTr->upper0 = *it->upper1;
                        it->upper1 = nullptr;
                        rightTr->upper1 = *it->usave;

                        it->upper0->lower0 = it;
                        rightTr->upper0->lower0 = rightTr;
                        rightTr->upper1->lower0 = rightTr;
                    }
                    else {

                        rightTr->upper1 = nullptr;
                        rightTr->upper0 = *it->upper1;
                        it->upper1 = *it->upper0;
                        it->upper0 = *it->usave;

                        it->upper0->lower0 = it;
                        it->upper1->lower0 = it;
                        rightTr->upper0->lower0 = rightTr;
                    }

                    it->usave = nullptr;
                    rightTr->usave = nullptr;
                }
                else {

                    rightTr->upper0 = it->upper1;
                    it->upper1 = nullptr;
                    rightTr->upper1 = nullptr;
                    rightTr->upper0->lower0 = rightTr;
                }
            }
            else {

                auto tmpU = it->upper0;

                if(*tmpU->lower0 && *tmpU->lower1) {

                    auto td0 = *tmpU->lower0;
                    auto td1 = *tmpU->lower1;
                    if(*td0->rightSegment && !IsLeftOf(*td0->rightSegment, seg->v1)) {
                        it->upper0 = nullptr;
                        it->upper1 = nullptr;
                        rightTr->upper0->lower1 = rightTr;
                    }
                    else {

                        rightTr->upper0 = nullptr;
                        rightTr->upper1 = nullptr;
                        it->upper0->lower0 = it;
                    }
                }
                else {
                    it->upper0->lower0 = it;
                    rightTr->upper0->lower1 = rightTr;
                }
            }


            if(FP_EQUAL(it->low.y, lastTrapezoid->low.y) && FP_EQUAL(it->low.x, lastTrapezoid->low.x) && triBottom) {

                if(swapped) {
                    tmpTriSeg = *seg->prev;
                }
                else {
                    tmpTriSeg = *seg->next;
                }

                if(tmpTriSeg && IsLeftOf(tmpTriSeg, seg->v0)) {

                    it->lower0->upper0 = it;
                    rightTr->lower0 = nullptr;
                    rightTr->lower1 = nullptr;
                }
                else {

                    rightTr->lower0->upper1 = rightTr;
                    it->lower0 = nullptr;
                    it->lower1 = nullptr;
                }
            }
            else {

                if(*it->lower0->upper0 && *it->lower0->upper1) {

                    if(*it->lower0->upper0 == it) {
                        it->lower0->usave = *it->lower0->upper1;
                        it->lower0->uside = S_LEFT;
                    }
                    else {
                        it->lower0->usave = *it->lower0->upper0;
                        it->lower0->uside = S_RIGHT;
                    }
                }
                it->lower0->upper0 = it;
                it->lower0->upper1 = rightTr;
            }

            it = *it->lower0;
        }
        else if(*it->lower0 && *it->lower1) {

            if(*it->upper0 && *it->upper1) {

                if(*it->usave) {

                    if(it->uside == S_LEFT) {

                        rightTr->upper0 = *it->upper1;
                        it->upper1 = nullptr;
                        rightTr->upper1 = *it->usave;

                        it->upper0->lower0 = it;
                        rightTr->upper0->lower0 = rightTr;
                        rightTr->upper1->lower0 = rightTr;
                    }
                    else {

                        rightTr->upper1 = nullptr;
                        rightTr->upper0 = *it->upper1;
                        it->upper1 = *it->upper0;
                        it->upper0 = *it->usave;

                        it->upper0->lower0 = it;
                        it->upper1->lower0 = it;
                        rightTr->upper0->lower0 = rightTr;
                    }

                    it->usave = nullptr;
                    rightTr->usave = nullptr;
                }
                else {
                    rightTr->upper0 = *it->upper1;
                    it->upper1 = nullptr;
                    rightTr->upper1 = nullptr;
                    rightTr->upper0->lower0 = rightTr;
                }
            }
            else {

                if(*it->upper0->lower0 && *it->upper0->lower1) {

                    if(*it->upper0->lower0->rightSegment && !IsLeftOf(*it->upper0->lower0->rightSegment, seg->v1)) {

                        it->upper0 = nullptr;
                        it->upper1 = nullptr;
                        rightTr->upper1 = nullptr;
                        rightTr->upper0->lower1 = rightTr;
                    }
                    else {
                        rightTr->upper0 = nullptr;
                        rightTr->upper1 = nullptr;
                        it->upper1 = nullptr;
                        it->upper0->lower0 = it;
                    }
                }
                else {

                    it->upper0->lower0 = it;
                    it->upper0->lower1 = rightTr;
                }
            }

            if(FP_EQUAL(it->low.y, lastTrapezoid->low.y) && FP_EQUAL(it->low.x, lastTrapezoid->low.x) && triBottom) {

                SegmentQ* tmpSeg{};

                if(swapped) {
                    tmpTriSeg = *seg->prev;
                }
                else {
                    tmpTriSeg = *seg->next;
                }

                if(tmpSeg && IsLeftOf(tmpSeg, seg->v0)) {

                    it->lower1->upper0 = it;
                    rightTr->lower0 = nullptr;
                    rightTr->lower1 = nullptr;
                }
                else {

                    rightTr->lower1->upper1 = rightTr;
                    it->lower0 = nullptr;
                    it->lower1 = nullptr;
                }
            }
            else {

                if(*it->lower1->upper0 && *it->lower1->upper1) {

                    if(*it->lower1->upper0 == it) {

                        it->lower1->usave = *it->lower1->upper1;
                        it->lower1->uside = S_LEFT;
                    }
                    else {

                        it->lower1->usave = *it->lower1->upper0;
                        it->lower1->uside = S_RIGHT;
                    }
                }

                it->lower1->upper0 = it;
                it->lower1->upper1 = rightTr;
            }


            it = *it->lower1;
        }
        else {

            auto tmpSeg = *it->lower0->rightSegment;
            int i_id0 = false;
            int i_id1 = false;
            f32 y0;
            f32 yt;
            v2 tmpPt;
            TrapezoidQ* tnext;

            if(FP_EQUAL(it->low.y, seg->v0.y)) {

                if(it->low.x > seg->v0.x) {

                    i_id0 = true;
                }
                else {

                    i_id1 = true;
                }
            }
            else {

                tmpPt.y = it->low.y;
                y0 = it->low.y;

                yt = (y0 - seg->v0.y) / (seg->v1.y - seg->v0.y);
                tmpPt.x = seg->v0.x + yt * (seg->v1.x - seg->v0.x);

                if(LessThanPoint(tmpPt, it->low)) {

                    i_id0 = true;
                }
                else {
                    i_id1 = true;
                }
            }

            if(*it->upper1 && *it->upper1) {

                if(*it->usave) {

                    if(*it->usave) {

                        rightTr->upper0 = *it->upper1;
                        it->upper1 = nullptr;
                        rightTr->upper1 = *it->usave;

                        it->upper0->lower0 = it;
                        rightTr->upper0->lower0 = rightTr;
                        rightTr->upper1->lower0 = rightTr;
                    }
                    else {
                        rightTr->upper0 = *it->upper1;
                        rightTr->upper1 = nullptr;
                        it->upper1 = nullptr;
                        rightTr->upper0->lower0 = rightTr;
                    }

                    it->usave = nullptr;
                    rightTr->usave = nullptr;
                }
                else {

                    rightTr->upper0 = *it->upper1;
                    rightTr->upper1 = nullptr;
                    it->upper1 = nullptr;
                    rightTr->upper0->lower0 = rightTr;
                }
            }
            else {

                if(*it->upper0->lower0 && *it->upper0->lower1) {

                    if(*it->upper0->lower0->rightSegment && !IsLeftOf(*it->upper0->lower0->rightSegment, seg->v1)) {

                        it->upper0 = nullptr;
                        it->upper1 = nullptr;
                        rightTr->upper1 = nullptr;
                        rightTr->upper0->lower1 = rightTr;
                    }
                    else {

                        rightTr->upper0 = nullptr;
                        rightTr->upper1 = nullptr;
                        it->upper1 = nullptr;
                        it->upper0->lower0 = it;
                    }
                }
                else {

                    it->upper0->lower0 = it;
                    it->upper0->lower1 = rightTr;
                }
            }

            if(FP_EQUAL(it->low.y, lastTrapezoid->low.y) && FP_EQUAL(it->low.x, lastTrapezoid->low.x) && triBottom ) {


                it->lower0->upper0 = it;
                it->lower0->upper1 = nullptr;
                it->lower1->upper0 = rightTr;
                it->lower1->upper1 = nullptr;

                rightTr->lower0 = *it->lower1;
                it->lower1 = nullptr;
                rightTr->lower1 = nullptr;

                tnext = *it->lower0;
            }
            else if(i_id0) {

                it->lower0->upper0 = it;
                it->lower0->upper1 = rightTr;
                it->lower1->upper0 = rightTr;
                it->lower1->upper1 = nullptr;

                rightTr->lower0 = *it->upper1;
                it->upper1 = nullptr;
                rightTr->lower1 = nullptr;

                tnext = *it->lower1;
            }
            else {

                it->lower0->upper0 = it;
                it->lower0->upper1 = nullptr;
                it->lower1->upper0 = it;
                it->lower1->upper1 = rightTr;

                rightTr->lower0 = *it->lower1;
                rightTr->lower1 = nullptr;

                tnext = *it->lower1;
            }

            it = tnext;
        }

        t_sav->rightSegment = seg;
        tn_sav->leftSegment = seg;
    }


    firstLeftTr = firstTrapezoid;
    lastLeftTr = lastTrapezoid;

    MergeTrapezoids(seg, firstLeftTr, lastLeftTr, S_LEFT);
    MergeTrapezoids(seg, firstRightTr, lastRightTr, S_RIGHT);
    seg->is_inserted = true;
}



void SegmentFindNewRoot(SegmentQ* seg) {

    if (seg->is_inserted) {
        return;
    }
    seg->root0 = *(LocateTrapezoid(*seg->root0, seg->v0, seg->v1)->sink);
    seg->root1 = *(LocateTrapezoid(*seg->root1, seg->v0, seg->v1)->sink);
}
QNode* BuildQueryStructure(u32 segCount, SegmentQ* segments, MemoryPool<sizeof(TrapezoidQ)>* trPool, MemoryPool<sizeof(QNode)>* qPool) {

    SegmentQ* s = segments;

    memset(trPool->base, 0, trPool->poolSize);
    memset(qPool->base, 0, qPool->poolSize);

    // i1
    auto root = (QNode*)pool_allocate(qPool);
    *root = {};

    root->nodetype = Q_Y_NODE;
    root->yVal = MaxPoint(s->v0, s->v1);

    // i2
    auto rootRight = (QNode*)pool_allocate(qPool);
    *rootRight = {};
    root->right = rootRight;
    rootRight->nodetype = Q_SINK_NODE;
    rootRight->parent = root;

    // i3
    auto rootLeft = (QNode*)pool_allocate(qPool);
    *rootLeft = {};
    root->left = rootLeft;
    rootLeft->nodetype = Q_Y_NODE;
    rootLeft->yVal = MinPoint(s->v0, s->v1);
    rootLeft->parent = root;

    // i4
    auto rootLeftLeft = (QNode*)pool_allocate(qPool);
    *rootLeftLeft = {};
    rootLeft->left = rootLeftLeft;
    rootLeftLeft->nodetype = Q_SINK_NODE;
    rootLeftLeft->parent = rootLeft;

    // i5
    auto rootLeftRight = (QNode*)pool_allocate(qPool);
    *rootLeftRight = {};
    rootLeft->right = rootLeftRight;
    rootLeftRight->nodetype = Q_X_NODE;
    rootLeftRight->segment = s;
    rootLeftRight->parent = rootLeft;

    // i6
    auto rootLeftRightLeft = (QNode*)pool_allocate(qPool);
    *rootLeftRightLeft = {};
    rootLeftRight->left = rootLeftRightLeft;
    rootLeftRightLeft->nodetype = Q_SINK_NODE;
    rootLeftRightLeft->parent = rootLeftRight;

    // i7
    auto rootLeftRightLeftRight = (QNode*)pool_allocate(qPool);
    *rootLeftRightLeftRight = {};
    rootLeftRight->right = rootLeftRightLeftRight;
    rootLeftRightLeftRight->nodetype = Q_SINK_NODE;
    rootLeftRightLeftRight->parent = rootLeftRight;

    auto t1 = (TrapezoidQ*)pool_allocate(trPool);
    auto t2 = (TrapezoidQ*)pool_allocate(trPool);
    auto t3 = (TrapezoidQ*)pool_allocate(trPool);
    auto t4 = (TrapezoidQ*)pool_allocate(trPool);
    *t1 = {};
    *t2 = {};
    *t3 = {};
    *t4 = {};

    t1->low = root->yVal;
    t2->low = root->yVal;
    t3->high = root->yVal;
    t4->high = {
        (f32)INFINITY,
        (f32)INFINITY
    };
    t3->low = {
        - (f32)INFINITY,
        - (f32)INFINITY
    };
    t1->rightSegment = s;
    t2->leftSegment = s;

    t1->lower0 = t3;
    t1->upper0 = t4;

    t2->lower0 = t3;
    t2->upper0 = t4;

    t3->upper0 = t1;
    t3->upper1 = t2;

    t4->lower0 = t1;
    t4->lower1 = t2;

    t1->sink = rootLeftRightLeft;
    t2->sink = rootLeftRightLeftRight;
    t3->sink = rootLeftLeft;
    t4->sink = rootRight;

    t1->state = TRAPEZOID_VALID;
    t2->state = TRAPEZOID_VALID;
    t3->state = TRAPEZOID_VALID;
    t4->state = TRAPEZOID_VALID;

    rootLeftRightLeft->trapezoid = t1;
    rootLeftRightLeftRight->trapezoid = t2;
    rootLeftLeft->trapezoid = t3;
    rootRight->trapezoid = t4;

    s->is_inserted = true;

    for(u32 i = 1; i < segCount; i++) {
        segments[i].root0 = root;
        segments[i].root1 = root;
  	}

    u32 segmentI = 1;
    for(u32 h = 1; h < math_logstar_n(segCount); h++) {
        for (u32 i = math_N(segCount, h -1) + 1; i <= math_N(segCount, h); i++) {
            AddSegment(root, segments + (segmentI++), trPool, qPool);
        }
    }

    // Find a new root for each of the segment endpoints
    for(u32 i = 1; i <= segCount; i++) {
        SegmentFindNewRoot(segments + i);
    }

    for(u32 i = math_N(segCount, math_logstar_n(segCount)) + 1; i <= segCount; i++) {
        AddSegment(root, segments + (segmentI++), trPool, qPool);
    }
}


u32 TrianulatePolygon(u32 vcount, v2* poly, LinearAllocator* dst) {

    SegmentQ segments[2];
    u32 half = (dst->cap - dst->top) >> 1;

    auto trPool = make_memory_pool<sizeof(TrapezoidQ)>(linear_allocate(dst, half), half);
    auto qPool  = make_memory_pool<sizeof(QNode)>(linear_allocate(dst, half), half);
    auto root = BuildQueryStructure(2, segments, &trPool, &qPool);


}
*/
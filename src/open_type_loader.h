#include <common.h>
#include <math3d.h>

#define OTF_TAG_STR(str) (((u32)str[3] << 24) | ((u32)str[2] << 16) | ((u32)str[1] << 8) | (u32) str[0] )
constexpr u32 OTF_TABLES[] = {

    OTF_TAG_STR("cmap"), //
    OTF_TAG_STR("loca"), //
    OTF_TAG_STR("head"), //
    OTF_TAG_STR("glyf"),
    OTF_TAG_STR("hhea"),
    OTF_TAG_STR("CFF "),
    OTF_TAG_STR("hmtx"),
    OTF_TAG_STR("kern"),
    OTF_TAG_STR("GPOS"),
};


typedef char TagOTF[4];
struct __attribute__ ((packed)) TableRecordOTF {

    TagOTF tableTag;
    u32 checkSum;
    u32 offset;
    u32 len;
};
struct __attribute__ ((packed)) TableDirectoryOTF {

    u32 sfntVersion;
    u16 tableCount;
    u16 searchRange;
    u16 entrySelector;
    u16 rangeShift;
    TableRecordOTF tableRecords[];
};

struct OTFTableInfo {

    TagOTF tableTag;
    u32 checkSum;
    u32 offset;
    u32 len;
};
struct OTFCodePointSegment {
    u32 start;
    u32 end;
    u16 offset;
    i16 delta;
};

struct __attribute__ ((packed)) SequentialMapGroupOTF {
    u32 startCharCode;
    u32 endCharCode;
    u32 startGlyphID;
};
struct __attribute__ ((packed)) ConstantMapGroupOTF {
    u32 startCharCode;
    u32 endCharCode;
    u32 glyphID;
};
struct OTFDenseCodePointMap {
    bool sequential;
    u32 groupCount;
    SequentialMapGroupOTF* groups;
};
struct OTFSparseCodePointMap {
    u32 segmentCount;
    u32 glyphIDCount;
    OTFCodePointSegment* segments;
    u16* glyphIDs;
};

struct __attribute__ ((packed)) HeadTableOTF {

    u16 majorVersion;
    u16 minorVersion;
    u32 fixedFontRevision;
    u32 checksumAdjustment;

    u32 magicNum; // 0x5F0F3CF5
    u16 flags;
    u16 unitsPerEm;

    i64 created;
    i64 modified;

    i16 xMin;
    i16 yMin;
    i16 xMax;
    i16 yMax;

    u16 macStyle;
    u16 lowestRecPPEM;
    i16 fontDirectionHint;
    i16 locaFormat;
    i16 glyphDataFormat;
};
struct CFFIndexEntry {
    byte* begin;
    byte* end;
};
struct DictionaryCFF {

    byte* begin;
    byte* end;
    u8*   it;
};
struct __attribute__ ((packed)) IndexCFF {

    u16     count;
    u8      offSize;
    byte    offsets[/* count + 1 */];
    // u8 data[];
};
struct __attribute__ ((packed)) HeaderCFF {

    u8 major;
    u8 minor;
    u8 hdrSize;
    u8 offSize;
};
struct CFFInfo {

    HeaderCFF head;
    IndexCFF* nameIndex;
    IndexCFF* topDictIndex;
    IndexCFF* stringIndex;
    IndexCFF* globalSubroutineIndex;
    IndexCFF* subroutineIndex;
    IndexCFF* fontDictIndex;
    IndexCFF* charStringIndex;

    DictionaryCFF privateDict;
};
struct OTFInfo {

    byte* begin;
    TableRecordOTF* tableRecords;
    u32 sfntVersion;
    u16 tableCount;
    u16 searchRange;
    u16 entrySelector;
    u16 rangeShift;

    bool sparse; // 1 -> sparse, 0 -> dense
    union {
        OTFDenseCodePointMap    denseMap;
        OTFSparseCodePointMap   sparseMap;
    };

    union {
        TableRecordOTF* tables[SIZE_OF_ARRAY(OTF_TABLES)];
        struct {
            TableRecordOTF* cmap;
            TableRecordOTF* loca;
            TableRecordOTF* head;
            TableRecordOTF* glyf;
            TableRecordOTF* hhea;
            TableRecordOTF* cff ;
            TableRecordOTF* hmtx;
            TableRecordOTF* kern;
            TableRecordOTF* GPOS;
        };
    };

    HeadTableOTF headTable;
    CFFInfo* cffInfo;
};

enum PlatformIDOTF {
    OTF_PLATFORM_UNICODE,
    OTF_PLATFORM_MAC,
    OTF_PLATFORM_ISO,
    OTF_PLATFORM_WINDOWS,
    OTF_PLATFORM_CUSTOM,
};
enum UnicodePlatformIDOTF {
    OTF_UNICODE_PLATFORM_2_0_BMP = 3, // format 4 6
    OTF_UNICODE_PLATFORM_2_0_FULL,    // format 10 12
    OTF_UNICODE_PLATFORM_VAR_SEQ,     // format 14
    OTF_UNICODE_PLATFORM_FULL,        // format 13
};
struct __attribute__ ((packed)) Format4OTFHeader {

    u16 format; // ASSERT(format == 4);
    u16 length;
    u16 language;
    u16 segCountX2;
    u16 searchRange;
    u16 entrySelector;
    u16 rangeShift;
    u16 endCode[/* segCount */];
};
struct __attribute__ ((packed)) Format4OTFBody0 {
    u16 pad;
    u16 startCode[/* segCount */];
};
struct __attribute__ ((packed)) Format4OTFBody1 {
    u16 lastStartCode;
    i16 idDelta[/* segCount */];
};
struct __attribute__ ((packed)) Format4OTFBody2 {
    u16 lastIDDelta;
    i16 idRangeOffsets[/* segCount */];
};
struct __attribute__ ((packed)) Format4OTFBody3 {
    u16 lastIDRangeOffsets;
    u16 glyphID[/* arbitrary */];
};
struct __attribute__ ((packed)) Format6OTF {
    u16 format;  // ASSERT(format == 6);
    u16 length;
    u16 language;
    u16 firstCode;
    u16 entryCount;
    u16 glyphIDs[/* entryCount */];
};
struct __attribute__ ((packed)) Format10OTF {
    u16 format;     // ASSERT(format == 10);
    u16 reserved;   // ASSERT(reserved == 0);
    u32 length;
    u32 language;
    u32 startCharCode;
    u32 numChars;
    u16 glyphIDs[];
};

struct __attribute__ ((packed)) Format12OTF {
    u16 format;     // ASSERT(format == 12);
    u16 reserved;   // ASSERT(reserved == 0);
    u32 length;
    u32 language;
    u32 numGroups;
    SequentialMapGroupOTF groups[/* numGroups */];
};
struct __attribute__ ((packed)) Format13OTF {
    u16 format;     // ASSERT(format == 13);
    u16 reserved;   // ASSERT(reserved == 0);
    u32 length;
    u32 language;
    u32 numGroups;
    ConstantMapGroupOTF groups[/* numGroups */];
};
struct __attribute__ ((packed)) VariationSelectorOTF {

    typedef u8 u24[3];
    u24 varSelector;
    u32 defaultUVSOffset;
    u32 nonDefaultUVSOffset;
};
struct __attribute__ ((packed)) Format14OTF {
    u16 format;     // ASSERT(format == 14);
    u32 length;
    u32 numVarSelectorRecords;
    VariationSelectorOTF varSelector[/* numVarSelectorRecords */];
};

struct __attribute__ ((packed)) cmapSubTableOTF {

    u16 firstCode;
    u16 entryCount;
    u16 idDelta;
    u16 idRangeOffset;
};
struct __attribute__ ((packed)) EncodingRecordOTF {

    u16 platformID;
    u16 encodingID;
    u32 subTableOffset;
};

typedef u16* locaU16TableOTF;
typedef u32* locaU32TableOTF;

struct __attribute__ ((packed)) cmapTableOTF {

    u16 version;
    u16 numTables;
    EncodingRecordOTF encodings[];
};
struct __attribute__ ((packed)) maxp0_5TableOTF {
    u32 version; // 0x00005000
    u16 numGlyphs;
};
struct __attribute__ ((packed)) maxp1_0TableOTF {
    u32 version; // 0x00010000
    u16 numGlyphs;
};
struct __attribute__ ((packed)) glyfTableOTF {


};

enum VertexType : u32 {
    GLYPH_MOVE_TO,
    GLYPH_VERTEX_POINT,
    GLYPH_VERTEX_QUAD_BEZIER,
    GLYPH_VERTEX_CUBIC_BEZIER,
};
struct GylphVertexPoint {
    u32 type;
    v2 p;
};
struct GylphVertexQuadraticBezier {
    u32 type;
    v2 c0;
    v2 c1;
};
struct GylphVertexCubicBezier {
    u32 type;
    v2 c0;
    v2 c1;
    v2 c2;
};
struct GlyphOutline {
    u32 vertexCount;
    u32 contourCount;
    byte* begin;
    byte* end;
};

struct Edge {
    v2 p0;
    v2 p1;
};

OTFInfo ParseOTFMemory(byte* mem, u32 size, LinearAllocator* alloc);
GlyphOutline ExtractGlyphOutlineOTF(OTFInfo* info, u32 codePoint, LinearAllocator* res);
u32 MakeEdgelistFromOutline(GlyphOutline outline, f32 flattnessThreshold, u32 contours[], Edge** res, LinearAllocator* alloc);
u32 ComputeTriangles(GlyphOutline outline, v2 scale, v2 translate, LinearAllocator* dst);
f32 GetScaleForPixelHeight(OTFInfo* info, f32 height);
u32 Unwind(v2* points, u32* wcount, u32 windings, v2 offset, bool invert, LinearAllocator* alloc);
u32 TesselateOutline(GlyphOutline outline, u32 contourPointCount[], f32 flattnessThreshold, LinearAllocator* alloc);
u32 TriangulateTesselatedContours(i32 (*dst)[3], u32 contourCount, u32 contourPointCount[], u32 pCount, v2* src, LinearAllocator* alloc);
v2 RayLineSegmentIntersection(v2 rayOrigin, v2 rayDir, v2 a, v2 b);
v2 LineLineSegmentIntersection(v2 rayOrigin, v2 rayDir, v2 a, v2 b);
void QsortPointsX(v2* arr, i64 low, i64 high);
void QsortPointsY(v2* arr, i64 low, i64 high);

u32 TriangulatePolygon(u32 contourCount, u32 contours[], v2* poly, LinearAllocator* dst);

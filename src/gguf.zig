// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const Error = error{
    InvalidGguf,
    UnsupportedGgufVersion,
    UnsupportedMetadataType,
    UnsupportedTensorType,
};

pub const default_alignment: usize = 32;

pub const MetadataType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,

    pub fn isNativeFloat(self: TensorType) bool {
        return switch (self) {
            .f32, .f16, .bf16 => true,
            else => false,
        };
    }
};

pub const ArrayInfo = struct {
    item_type: MetadataType,
    len: usize,
    values_offset: usize,
};

pub const MetadataValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: ArrayInfo,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const Metadata = struct {
    key: []const u8,
    value: MetadataValue,
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: [8]u64,
    dim_count: u32,
    tensor_type: TensorType,
    offset: u64,

    pub fn shape(self: *const TensorInfo) []const u64 {
        return self.dims[0..self.dim_count];
    }

    pub fn elementCount(self: *const TensorInfo) usize {
        var total: usize = 1;
        for (self.shape()) |dim| total *= @intCast(dim);
        return total;
    }
};

pub const File = struct {
    allocator: Allocator,
    bytes: []const u8,
    version: u32,
    metadata: []Metadata,
    tensors: []TensorInfo,
    data_start: usize,
    alignment: usize,

    pub fn deinit(self: *File) void {
        self.allocator.free(self.metadata);
        self.allocator.free(self.tensors);
        self.* = undefined;
    }

    pub fn metadataValue(self: File, key: []const u8) ?MetadataValue {
        for (self.metadata) |item| {
            if (std.mem.eql(u8, item.key, key)) return item.value;
        }
        return null;
    }

    pub fn string(self: File, key: []const u8) ?[]const u8 {
        const value = self.metadataValue(key) orelse return null;
        return switch (value) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn usizeValue(self: File, key: []const u8) ?usize {
        const value = self.metadataValue(key) orelse return null;
        return switch (value) {
            .uint8 => |v| v,
            .uint16 => |v| v,
            .uint32 => |v| v,
            .uint64 => |v| @intCast(v),
            .int8 => |v| if (v >= 0) @intCast(v) else null,
            .int16 => |v| if (v >= 0) @intCast(v) else null,
            .int32 => |v| if (v >= 0) @intCast(v) else null,
            .int64 => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    pub fn f32Value(self: File, key: []const u8) ?f32 {
        const value = self.metadataValue(key) orelse return null;
        return switch (value) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            .uint8 => |v| @floatFromInt(v),
            .uint16 => |v| @floatFromInt(v),
            .uint32 => |v| @floatFromInt(v),
            .uint64 => |v| @floatFromInt(v),
            .int8 => |v| @floatFromInt(v),
            .int16 => |v| @floatFromInt(v),
            .int32 => |v| @floatFromInt(v),
            .int64 => |v| @floatFromInt(v),
            else => null,
        };
    }

    pub fn tensor(self: File, name: []const u8) ?TensorInfo {
        for (self.tensors) |info| {
            if (std.mem.eql(u8, info.name, name)) return info;
        }
        return null;
    }

    pub fn stringArrayAlloc(self: File, allocator: Allocator, key: []const u8) ![][]const u8 {
        const value = self.metadataValue(key) orelse return Error.InvalidGguf;
        const array = switch (value) {
            .array => |a| a,
            else => return Error.InvalidGguf,
        };
        if (array.item_type != .string) return Error.InvalidGguf;

        const out = try allocator.alloc([]const u8, array.len);
        errdefer allocator.free(out);

        var reader: Reader = .{ .bytes = self.bytes, .index = array.values_offset };
        for (out) |*item| item.* = try reader.readString();
        return out;
    }
};

pub fn parse(allocator: Allocator, bytes: []const u8) !File {
    var reader: Reader = .{ .bytes = bytes };
    if (!std.mem.eql(u8, try reader.readBytes(4), "GGUF")) return Error.InvalidGguf;

    const version = try reader.readInt(u32);
    if (version < 2 or version > 3) return Error.UnsupportedGgufVersion;

    const tensor_count: usize = @intCast(try reader.readInt(u64));
    const metadata_count: usize = @intCast(try reader.readInt(u64));

    const metadata = try allocator.alloc(Metadata, metadata_count);
    errdefer allocator.free(metadata);
    for (metadata) |*item| {
        const key = try reader.readString();
        const value_type = try reader.readEnum(MetadataType);
        item.* = .{ .key = key, .value = try reader.readMetadataValue(value_type) };
    }

    const tensors = try allocator.alloc(TensorInfo, tensor_count);
    errdefer allocator.free(tensors);
    for (tensors) |*info| {
        const name = try reader.readString();
        const dim_count = try reader.readInt(u32);
        if (dim_count == 0 or dim_count > 8) return Error.InvalidGguf;

        var dims: [8]u64 = @splat(0);
        for (0..dim_count) |i| dims[i] = try reader.readInt(u64);

        info.* = .{
            .name = name,
            .dims = dims,
            .dim_count = dim_count,
            .tensor_type = try reader.readEnum(TensorType),
            .offset = try reader.readInt(u64),
        };
    }

    var file = File{
        .allocator = allocator,
        .bytes = bytes,
        .version = version,
        .metadata = metadata,
        .tensors = tensors,
        .data_start = 0,
        .alignment = default_alignment,
    };
    file.alignment = file.usizeValue("general.alignment") orelse default_alignment;
    if (file.alignment == 0) return Error.InvalidGguf;
    file.data_start = std.mem.alignForward(usize, reader.index, file.alignment);
    if (file.data_start > bytes.len) return Error.InvalidGguf;
    return file;
}

const Reader = struct {
    bytes: []const u8,
    index: usize = 0,

    fn readBytes(self: *Reader, len: usize) ![]const u8 {
        if (self.index + len > self.bytes.len) return Error.InvalidGguf;
        const out = self.bytes[self.index..][0..len];
        self.index += len;
        return out;
    }

    fn readInt(self: *Reader, comptime T: type) !T {
        const raw = try self.readBytes(@sizeOf(T));
        return std.mem.readInt(T, raw[0..@sizeOf(T)], .little);
    }

    fn readFloat(self: *Reader, comptime T: type) !T {
        return @bitCast(try self.readInt(std.meta.Int(.unsigned, @bitSizeOf(T))));
    }

    fn readEnum(self: *Reader, comptime T: type) !T {
        return std.enums.fromInt(T, try self.readInt(u32)) orelse Error.UnsupportedMetadataType;
    }

    fn readString(self: *Reader) ![]const u8 {
        const len: usize = @intCast(try self.readInt(u64));
        return self.readBytes(len);
    }

    fn readMetadataValue(self: *Reader, value_type: MetadataType) !MetadataValue {
        return switch (value_type) {
            .uint8 => .{ .uint8 = try self.readInt(u8) },
            .int8 => .{ .int8 = try self.readInt(i8) },
            .uint16 => .{ .uint16 = try self.readInt(u16) },
            .int16 => .{ .int16 = try self.readInt(i16) },
            .uint32 => .{ .uint32 = try self.readInt(u32) },
            .int32 => .{ .int32 = try self.readInt(i32) },
            .float32 => .{ .float32 = try self.readFloat(f32) },
            .bool => .{ .bool = (try self.readInt(u8)) != 0 },
            .string => .{ .string = try self.readString() },
            .array => blk: {
                const item_type = try self.readEnum(MetadataType);
                const len: usize = @intCast(try self.readInt(u64));
                const values_offset = self.index;
                for (0..len) |_| try self.skipMetadataValue(item_type);
                break :blk .{ .array = .{ .item_type = item_type, .len = len, .values_offset = values_offset } };
            },
            .uint64 => .{ .uint64 = try self.readInt(u64) },
            .int64 => .{ .int64 = try self.readInt(i64) },
            .float64 => .{ .float64 = try self.readFloat(f64) },
        };
    }

    fn skipMetadataValue(self: *Reader, value_type: MetadataType) !void {
        switch (value_type) {
            .uint8, .int8, .bool => _ = try self.readBytes(1),
            .uint16, .int16 => _ = try self.readBytes(2),
            .uint32, .int32, .float32 => _ = try self.readBytes(4),
            .uint64, .int64, .float64 => _ = try self.readBytes(8),
            .string => _ = try self.readString(),
            .array => {
                const item_type = try self.readEnum(MetadataType);
                const len: usize = @intCast(try self.readInt(u64));
                for (0..len) |_| try self.skipMetadataValue(item_type);
            },
        }
    }
};

test "parses gguf metadata and tensor table" {
    const allocator = std.testing.allocator;
    var bytes = std.array_list.Managed(u8).init(allocator);
    defer bytes.deinit();

    try writeBytes(&bytes, "GGUF");
    try writeInt(&bytes, u32, 3);
    try writeInt(&bytes, u64, 1);
    try writeInt(&bytes, u64, 4);

    try writeString(&bytes, "general.architecture");
    try writeInt(&bytes, u32, @intFromEnum(MetadataType.string));
    try writeString(&bytes, "llama");

    try writeString(&bytes, "llama.block_count");
    try writeInt(&bytes, u32, @intFromEnum(MetadataType.uint32));
    try writeInt(&bytes, u32, 16);

    try writeString(&bytes, "general.alignment");
    try writeInt(&bytes, u32, @intFromEnum(MetadataType.uint32));
    try writeInt(&bytes, u32, 32);

    try writeString(&bytes, "tokenizer.ggml.tokens");
    try writeInt(&bytes, u32, @intFromEnum(MetadataType.array));
    try writeInt(&bytes, u32, @intFromEnum(MetadataType.string));
    try writeInt(&bytes, u64, 2);
    try writeString(&bytes, "hello");
    try writeString(&bytes, "world");

    try writeString(&bytes, "token_embd.weight");
    try writeInt(&bytes, u32, 2);
    try writeInt(&bytes, u64, 4);
    try writeInt(&bytes, u64, 8);
    try writeInt(&bytes, u32, @intFromEnum(TensorType.f32));
    try writeInt(&bytes, u64, 0);
    while (bytes.items.len % 32 != 0) try bytes.append(0);

    var parsed = try parse(allocator, bytes.items);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u32, 3), parsed.version);
    try std.testing.expectEqualStrings("llama", parsed.string("general.architecture").?);
    try std.testing.expectEqual(@as(usize, 16), parsed.usizeValue("llama.block_count").?);
    try std.testing.expect(parsed.data_start % 32 == 0);

    const tensor = parsed.tensor("token_embd.weight").?;
    try std.testing.expectEqual(TensorType.f32, tensor.tensor_type);
    try std.testing.expectEqual(@as(usize, 32), tensor.elementCount());

    const tokens = try parsed.stringArrayAlloc(allocator, "tokenizer.ggml.tokens");
    defer allocator.free(tokens);
    try std.testing.expectEqualStrings("hello", tokens[0]);
    try std.testing.expectEqualStrings("world", tokens[1]);
}

fn writeBytes(out: *std.array_list.Managed(u8), bytes: []const u8) !void {
    try out.appendSlice(bytes);
}

fn writeString(out: *std.array_list.Managed(u8), value: []const u8) !void {
    try writeInt(out, u64, value.len);
    try writeBytes(out, value);
}

fn writeInt(out: *std.array_list.Managed(u8), comptime T: type, value: T) !void {
    const start = out.items.len;
    try out.appendNTimes(0, @sizeOf(T));
    std.mem.writeInt(T, out.items[start..][0..@sizeOf(T)], value, .little);
}

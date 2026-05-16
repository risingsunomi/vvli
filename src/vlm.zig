// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const gguf = @import("gguf");
const vision = @import("vision");

pub const Allocator = std.mem.Allocator;

pub const Error = error{
    NativeVisionExecutionIncomplete,
    UnsupportedProjectorDType,
};

pub const NativePlan = struct {
    allocator: Allocator,
    image: vision.ImageInput,
    language_architecture: []u8,
    language_tensor_count: usize,
    projector_architecture: []u8,
    projector_type: []u8,
    has_vision_encoder: bool,
    image_size: ?usize,
    patch_size: ?usize,
    embedding_length: ?usize,
    projection_dim: ?usize,
    block_count: ?usize,
    projector_tensor_count: usize,
    projector_native_tensors: usize,
    projector_quantized_tensors: usize,

    pub fn deinit(self: *NativePlan) void {
        self.allocator.free(self.language_architecture);
        self.allocator.free(self.projector_architecture);
        self.allocator.free(self.projector_type);
        self.* = undefined;
    }
};

pub fn loadNativePlan(
    allocator: Allocator,
    io: std.Io,
    language_model_path: []const u8,
    projector_path: []const u8,
    image: vision.ImageInput,
) !NativePlan {
    var language_file = try std.Io.Dir.cwd().openFile(io, language_model_path, .{ .mode = .read_only });
    defer language_file.close(io);
    const language_stat = try language_file.stat(io);
    var language_map = try std.Io.File.MemoryMap.create(io, language_file, .{
        .len = @intCast(language_stat.size),
        .protection = .{ .read = true, .write = false },
        .populate = false,
    });
    defer language_map.destroy(io);

    var language_gguf = try gguf.parse(allocator, language_map.memory);
    defer language_gguf.deinit();

    var projector_file = try std.Io.Dir.cwd().openFile(io, projector_path, .{ .mode = .read_only });
    defer projector_file.close(io);
    const projector_stat = try projector_file.stat(io);
    var projector_map = try std.Io.File.MemoryMap.create(io, projector_file, .{
        .len = @intCast(projector_stat.size),
        .protection = .{ .read = true, .write = false },
        .populate = false,
    });
    defer projector_map.destroy(io);

    var projector_gguf = try gguf.parse(allocator, projector_map.memory);
    defer projector_gguf.deinit();

    const language_architecture = try dupMetadataString(allocator, language_gguf, "general.architecture", "unknown");
    errdefer allocator.free(language_architecture);

    const projector_architecture = try dupMetadataString(allocator, projector_gguf, "general.architecture", "clip");
    errdefer allocator.free(projector_architecture);

    const projector_type = try dupProjectorType(allocator, projector_gguf);
    errdefer allocator.free(projector_type);

    const projector_native_tensors = countNativeTensors(projector_gguf);
    const projector_tensor_count = projector_gguf.tensors.len;
    return .{
        .allocator = allocator,
        .image = image,
        .language_architecture = language_architecture,
        .language_tensor_count = language_gguf.tensors.len,
        .projector_architecture = projector_architecture,
        .projector_type = projector_type,
        .has_vision_encoder = optionalBool(projector_gguf, "clip.has_vision_encoder") orelse true,
        .image_size = projector_gguf.usizeValue("clip.vision.image_size"),
        .patch_size = projector_gguf.usizeValue("clip.vision.patch_size"),
        .embedding_length = projector_gguf.usizeValue("clip.vision.embedding_length"),
        .projection_dim = projector_gguf.usizeValue("clip.vision.projection_dim"),
        .block_count = projector_gguf.usizeValue("clip.vision.block_count"),
        .projector_tensor_count = projector_tensor_count,
        .projector_native_tensors = projector_native_tensors,
        .projector_quantized_tensors = projector_tensor_count - projector_native_tensors,
    };
}

fn dupProjectorType(allocator: Allocator, file: gguf.File) ![]u8 {
    const keys = [_][]const u8{
        "clip.vision.projector_type",
        "clip.projector_type",
    };
    for (keys) |key| {
        if (file.string(key)) |value| return allocator.dupe(u8, value);
    }
    return allocator.dupe(u8, "unknown");
}

fn dupMetadataString(allocator: Allocator, file: gguf.File, key: []const u8, fallback: []const u8) ![]u8 {
    return allocator.dupe(u8, file.string(key) orelse fallback);
}

fn optionalBool(file: gguf.File, key: []const u8) ?bool {
    const value = file.metadataValue(key) orelse return null;
    return switch (value) {
        .bool => |v| v,
        else => null,
    };
}

fn countNativeTensors(file: gguf.File) usize {
    var count: usize = 0;
    for (file.tensors) |tensor| {
        if (tensor.tensor_type.isNativeFloat()) count += 1;
    }
    return count;
}

test "native plan owns duplicated metadata strings" {
    const allocator = std.testing.allocator;
    var plan = NativePlan{
        .allocator = allocator,
        .image = .{ .path = "image.jpg", .byte_len = 3, .format = .jpeg },
        .language_architecture = try allocator.dupe(u8, "qwen35"),
        .language_tensor_count = 1,
        .projector_architecture = try allocator.dupe(u8, "clip"),
        .projector_type = try allocator.dupe(u8, "qwen3vl_merger"),
        .has_vision_encoder = true,
        .image_size = 512,
        .patch_size = 16,
        .embedding_length = 1024,
        .projection_dim = 2048,
        .block_count = 24,
        .projector_tensor_count = 2,
        .projector_native_tensors = 2,
        .projector_quantized_tensors = 0,
    };
    plan.deinit();
}

// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const builtin = @import("builtin");

const has_imageio = builtin.os.tag == .macos;
const c = if (has_imageio) struct {
    const CFURL = opaque {};
    const CGImageSource = opaque {};
    const CGImage = opaque {};
    const CGColorSpace = opaque {};
    const CGContext = opaque {};

    const CFURLRef = *CFURL;
    const CGImageSourceRef = *CGImageSource;
    const CGImageRef = *CGImage;
    const CGColorSpaceRef = *CGColorSpace;
    const CGContextRef = *CGContext;
    const CGFloat = f64;

    const CGPoint = extern struct {
        x: CGFloat,
        y: CGFloat,
    };
    const CGSize = extern struct {
        width: CGFloat,
        height: CGFloat,
    };
    const CGRect = extern struct {
        origin: CGPoint,
        size: CGSize,
    };

    const kCGImageAlphaPremultipliedLast: u32 = 1;
    const kCGBitmapByteOrder32Big: u32 = 0x00004000;
    const kCGInterpolationHigh: i32 = 3;

    extern "c" fn CFURLCreateFromFileSystemRepresentation(allocator: ?*const anyopaque, buffer: [*]const u8, bufLen: isize, isDirectory: u8) ?CFURLRef;
    extern "c" fn CFRelease(cf: *const anyopaque) void;
    extern "c" fn CGImageSourceCreateWithURL(url: CFURLRef, options: ?*const anyopaque) ?CGImageSourceRef;
    extern "c" fn CGImageSourceCreateImageAtIndex(isrc: CGImageSourceRef, index: usize, options: ?*const anyopaque) ?CGImageRef;
    extern "c" fn CGImageRelease(image: CGImageRef) void;
    extern "c" fn CGColorSpaceCreateDeviceRGB() ?CGColorSpaceRef;
    extern "c" fn CGColorSpaceRelease(space: CGColorSpaceRef) void;
    extern "c" fn CGBitmapContextCreate(data: ?*anyopaque, width: usize, height: usize, bitsPerComponent: usize, bytesPerRow: usize, space: CGColorSpaceRef, bitmapInfo: u32) ?CGContextRef;
    extern "c" fn CGContextRelease(context: CGContextRef) void;
    extern "c" fn CGContextSetInterpolationQuality(context: CGContextRef, quality: i32) void;
    extern "c" fn CGContextDrawImage(context: CGContextRef, rect: CGRect, image: CGImageRef) void;
} else struct {};

pub const Error = error{
    EmptyImageFile,
    InvalidImageFile,
    NativeImageDecodeUnsupported,
    UnsupportedImageFormat,
};

pub const Format = enum {
    jpeg,
    png,
    webp,
    bmp,
};

pub const ImageInput = struct {
    path: []const u8,
    byte_len: u64,
    format: Format,
};

pub const RgbImage = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    data: []align(64) f32,

    pub fn deinit(self: *RgbImage) void {
        if (self.data.len != 0) self.allocator.free(self.data);
        self.* = .{ .allocator = self.allocator, .width = 0, .height = 0, .data = emptyRgb() };
    }
};

pub fn inspectPath(io: std.Io, path: []const u8) !ImageInput {
    const format = detectFormat(path) orelse return Error.UnsupportedImageFormat;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    if (stat.size == 0) return Error.EmptyImageFile;

    var header: [12]u8 = undefined;
    var reader = file.reader(io, &.{});
    const header_len = try reader.interface.readSliceShort(&header);
    if (!matchesMagic(format, header[0..header_len])) return Error.InvalidImageFile;

    return .{
        .path = path,
        .byte_len = stat.size,
        .format = format,
    };
}

pub fn decodeResizeRgbF32(
    allocator: std.mem.Allocator,
    input: ImageInput,
    target_width: usize,
    target_height: usize,
    mean: [3]f32,
    stddev: [3]f32,
) !RgbImage {
    if (target_width == 0 or target_height == 0) return Error.InvalidImageFile;
    if (!has_imageio) return Error.NativeImageDecodeUnsupported;
    return decodeResizeRgbF32ImageIo(allocator, input, target_width, target_height, mean, stddev);
}

fn decodeResizeRgbF32ImageIo(
    allocator: std.mem.Allocator,
    input: ImageInput,
    target_width: usize,
    target_height: usize,
    mean: [3]f32,
    stddev: [3]f32,
) !RgbImage {
    const url = c.CFURLCreateFromFileSystemRepresentation(
        null,
        input.path.ptr,
        @intCast(input.path.len),
        0,
    ) orelse return Error.InvalidImageFile;
    defer c.CFRelease(@ptrCast(url));

    const source = c.CGImageSourceCreateWithURL(url, null) orelse return Error.InvalidImageFile;
    defer c.CFRelease(@ptrCast(source));

    const image = c.CGImageSourceCreateImageAtIndex(source, 0, null) orelse return Error.InvalidImageFile;
    defer c.CGImageRelease(image);

    const rgba_len = target_width * target_height * 4;
    const rgba = try allocator.alloc(u8, rgba_len);
    defer allocator.free(rgba);

    const color_space = c.CGColorSpaceCreateDeviceRGB() orelse return Error.InvalidImageFile;
    defer c.CGColorSpaceRelease(color_space);

    const bitmap_info = c.kCGImageAlphaPremultipliedLast | c.kCGBitmapByteOrder32Big;
    const context = c.CGBitmapContextCreate(
        @ptrCast(rgba.ptr),
        target_width,
        target_height,
        8,
        target_width * 4,
        color_space,
        bitmap_info,
    ) orelse return Error.InvalidImageFile;
    defer c.CGContextRelease(context);

    c.CGContextSetInterpolationQuality(context, c.kCGInterpolationHigh);
    c.CGContextDrawImage(context, c.CGRect{
        .origin = .{ .x = 0, .y = 0 },
        .size = .{ .width = @floatFromInt(target_width), .height = @floatFromInt(target_height) },
    }, image);

    const rgb = try allocator.alignedAlloc(f32, .@"64", target_width * target_height * 3);
    errdefer allocator.free(rgb);

    var src: usize = 0;
    var dst: usize = 0;
    while (src < rgba.len) : ({
        src += 4;
        dst += 3;
    }) {
        const r = @as(f32, @floatFromInt(rgba[src + 0])) / 255.0;
        const g = @as(f32, @floatFromInt(rgba[src + 1])) / 255.0;
        const b = @as(f32, @floatFromInt(rgba[src + 2])) / 255.0;
        rgb[dst + 0] = (r - mean[0]) / stddev[0];
        rgb[dst + 1] = (g - mean[1]) / stddev[1];
        rgb[dst + 2] = (b - mean[2]) / stddev[2];
    }

    return .{
        .allocator = allocator,
        .width = target_width,
        .height = target_height,
        .data = rgb,
    };
}

pub fn detectFormat(path: []const u8) ?Format {
    if (endsWithIgnoreCase(path, ".jpg") or endsWithIgnoreCase(path, ".jpeg")) return .jpeg;
    if (endsWithIgnoreCase(path, ".png")) return .png;
    if (endsWithIgnoreCase(path, ".webp")) return .webp;
    if (endsWithIgnoreCase(path, ".bmp")) return .bmp;
    return null;
}

fn endsWithIgnoreCase(text: []const u8, suffix: []const u8) bool {
    if (suffix.len > text.len) return false;
    const offset = text.len - suffix.len;
    for (suffix, 0..) |expected, i| {
        if (asciiLower(text[offset + i]) != asciiLower(expected)) return false;
    }
    return true;
}

fn asciiLower(ch: u8) u8 {
    return if (ch >= 'A' and ch <= 'Z') ch + ('a' - 'A') else ch;
}

fn emptyRgb() []align(64) f32 {
    return @as([*]align(64) f32, @ptrFromInt(64))[0..0];
}

fn matchesMagic(format: Format, header: []const u8) bool {
    return switch (format) {
        .jpeg => header.len >= 3 and header[0] == 0xff and header[1] == 0xd8 and header[2] == 0xff,
        .png => std.mem.eql(u8, header, "\x89PNG\r\n\x1a\n") or
            (header.len >= 8 and std.mem.eql(u8, header[0..8], "\x89PNG\r\n\x1a\n")),
        .webp => header.len >= 12 and std.mem.eql(u8, header[0..4], "RIFF") and std.mem.eql(u8, header[8..12], "WEBP"),
        .bmp => header.len >= 2 and std.mem.eql(u8, header[0..2], "BM"),
    };
}

test "detects common image extensions without allocating" {
    try std.testing.expectEqual(Format.jpeg, detectFormat("input.JPG").?);
    try std.testing.expectEqual(Format.jpeg, detectFormat("input.jpeg").?);
    try std.testing.expectEqual(Format.png, detectFormat("input.PNG").?);
    try std.testing.expectEqual(Format.webp, detectFormat("input.webp").?);
    try std.testing.expectEqual(Format.bmp, detectFormat("input.BMP").?);
}

test "rejects unsupported image extensions" {
    try std.testing.expectEqual(@as(?Format, null), detectFormat("input.txt"));
}

test "matches image magic bytes" {
    try std.testing.expect(matchesMagic(.png, "\x89PNG\r\n\x1a\n"));
    try std.testing.expect(matchesMagic(.jpeg, "\xff\xd8\xff\xe0"));
    try std.testing.expect(matchesMagic(.webp, "RIFFxxxxWEBP"));
    try std.testing.expect(matchesMagic(.bmp, "BM"));
    try std.testing.expect(!matchesMagic(.png, "not png"));
}

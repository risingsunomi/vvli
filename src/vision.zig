// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub const Error = error{
    EmptyImageFile,
    InvalidImageFile,
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

fn asciiLower(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + ('a' - 'A') else c;
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

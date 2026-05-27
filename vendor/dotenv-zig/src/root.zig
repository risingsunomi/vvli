// Adapted from https://github.com/velikoss/dotenv-zig for Zig 0.16.
const std = @import("std");

const Allocator = std.mem.Allocator;
const StringMap = std.StringHashMapUnmanaged([]const u8);

pub const Env = @This();

vars: StringMap,
arena: std.heap.ArenaAllocator,
process_env: ?*const std.process.Environ.Map = null,

const Entry = struct {
    key: []const u8,
    val: []const u8,
};

fn parseLine(line_content: []const u8) !?Entry {
    const line = std.mem.trim(u8, line_content, " \t\r\n");
    if (line.len == 0 or line[0] == '#') return null;

    const separator = std.mem.indexOfScalar(u8, line, '=') orelse return null;
    const key = std.mem.trim(u8, line[0..separator], " \t\r\n");
    var val = std.mem.trim(u8, line[separator + 1 ..], " \t\r\n");
    if (val.len > 1 and (val[0] == '"' or val[0] == '\'')) {
        if (val[0] != val[val.len - 1]) return error.ValueMalformed;
        val = val[1 .. val.len - 1];
    }
    return .{ .key = key, .val = val };
}

fn parseEnvFileContent(alloc: Allocator, content: []const u8) !StringMap {
    var env_map: StringMap = .empty;
    var rest = content;
    while (rest.len > 0) {
        const newline = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
        const line = rest[0..newline];
        rest = if (newline == rest.len) rest[rest.len..] else rest[newline + 1 ..];

        if (try parseLine(line)) |entry| {
            const key = try alloc.dupe(u8, entry.key);
            const val = try alloc.dupe(u8, entry.val);
            try env_map.put(alloc, key, val);
        }
    }
    return env_map;
}

pub fn parse_key(key: []const u8, content: []const u8) !?[]const u8 {
    var rest = content;
    while (rest.len > 0) {
        const newline = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
        const line = rest[0..newline];
        rest = if (newline == rest.len) rest[rest.len..] else rest[newline + 1 ..];

        if (try parseLine(line)) |entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry.val;
        }
    }
    return null;
}

pub fn init(alloc: Allocator, file_content: ?[]const u8) !Env {
    var arena = std.heap.ArenaAllocator.init(alloc);
    const vars = if (file_content) |content|
        try parseEnvFileContent(arena.allocator(), content)
    else
        StringMap.empty;

    return .{
        .vars = vars,
        .arena = arena,
    };
}

pub fn initWithProcessEnv(
    alloc: Allocator,
    file_content: ?[]const u8,
    process_env: ?*const std.process.Environ.Map,
) !Env {
    var env = try init(alloc, file_content);
    env.process_env = process_env;
    return env;
}

pub fn initWithPathIo(
    alloc: Allocator,
    io: std.Io,
    path: []const u8,
    max_bytes: usize,
    process_env: ?*const std.process.Environ.Map,
) !Env {
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(max_bytes)) catch |err| switch (err) {
        error.FileNotFound => return try initWithProcessEnv(alloc, null, process_env),
        else => return err,
    };
    defer alloc.free(content);

    return try initWithProcessEnv(alloc, content, process_env);
}

pub fn initWithPath(alloc: Allocator, path: []const u8, max_bytes: usize, use_process_env: bool) !Env {
    _ = use_process_env;
    return initWithPathIo(alloc, std.options.debug_io, path, max_bytes, null);
}

pub fn deinit(env: *Env) void {
    env.arena.deinit();
}

pub fn get(self: *Env, key: []const u8) ?[]const u8 {
    if (self.vars.get(key)) |value| return value;

    const process_env = self.process_env orelse return null;
    const process_value = process_env.get(key) orelse return null;

    const alloc = self.arena.allocator();
    const key_copy = alloc.dupe(u8, key) catch return null;
    const value_copy = alloc.dupe(u8, process_value) catch return null;
    self.vars.put(alloc, key_copy, value_copy) catch return null;
    return value_copy;
}

pub fn getRequired(self: *Env, key: []const u8) ![]const u8 {
    return self.get(key) orelse error.MissingRequiredEnvVar;
}

pub fn getWithDefault(self: *Env, key: []const u8, default: []const u8) []const u8 {
    return self.get(key) orelse default;
}

pub fn parseU16(self: *Env, key: []const u8) !u16 {
    const val = try self.getRequired(key);
    return std.fmt.parseInt(u16, val, 10) catch error.InvalidEnvVar;
}

pub fn parseU16WithDefault(self: *Env, key: []const u8, default: u16) u16 {
    const val = self.get(key) orelse return default;
    return std.fmt.parseInt(u16, val, 10) catch default;
}

pub fn parseU32(self: *Env, key: []const u8) !u32 {
    const val = try self.getRequired(key);
    return std.fmt.parseInt(u32, val, 10) catch error.InvalidEnvVar;
}

pub fn parseU32WithDefault(self: *Env, key: []const u8, default: u32) u32 {
    const val = self.get(key) orelse return default;
    return std.fmt.parseInt(u32, val, 10) catch default;
}

pub fn parseU64(self: *Env, key: []const u8) !u64 {
    const val = try self.getRequired(key);
    return std.fmt.parseInt(u64, val, 10) catch error.InvalidEnvVar;
}

pub fn parseU64WithDefault(self: *Env, key: []const u8, default: u64) u64 {
    const val = self.get(key) orelse return default;
    return std.fmt.parseInt(u64, val, 10) catch default;
}

pub fn parseUsize(self: *Env, key: []const u8) !usize {
    const val = try self.getRequired(key);
    return std.fmt.parseInt(usize, val, 10) catch error.InvalidEnvVar;
}

pub fn parseUsizeWithDefault(self: *Env, key: []const u8, default: usize) usize {
    const val = self.get(key) orelse return default;
    return std.fmt.parseInt(usize, val, 10) catch default;
}

pub fn parseBool(self: *Env, key: []const u8) !bool {
    const val = try self.getRequired(key);
    if (std.ascii.eqlIgnoreCase(val, "true") or std.ascii.eqlIgnoreCase(val, "1")) return true;
    if (std.ascii.eqlIgnoreCase(val, "false") or std.ascii.eqlIgnoreCase(val, "0")) return false;
    return error.InvalidEnvVar;
}

pub fn parseBoolWithDefault(self: *Env, key: []const u8, default: bool) bool {
    const val = self.get(key) orelse return default;
    if (std.ascii.eqlIgnoreCase(val, "true") or std.ascii.eqlIgnoreCase(val, "1")) return true;
    if (std.ascii.eqlIgnoreCase(val, "false") or std.ascii.eqlIgnoreCase(val, "0")) return false;
    return default;
}

pub fn parseFloat(self: *Env, key: []const u8) !f32 {
    const val = try self.getRequired(key);
    return std.fmt.parseFloat(f32, val) catch error.InvalidEnvVar;
}

pub fn parseFloatWithDefault(self: *Env, key: []const u8, default: f32) f32 {
    const val = self.get(key) orelse return default;
    return std.fmt.parseFloat(f32, val) catch default;
}

pub fn parseDouble(self: *Env, key: []const u8) !f64 {
    const val = try self.getRequired(key);
    return std.fmt.parseFloat(f64, val) catch error.InvalidEnvVar;
}

pub fn parseDoubleWithDefault(self: *Env, key: []const u8, default: f64) f64 {
    const val = self.get(key) orelse return default;
    return std.fmt.parseFloat(f64, val) catch default;
}

test "parses dotenv content" {
    const content =
        \\# ignored
        \\password="mysecretpassword"
        \\number=123
        \\somekey=somekey
    ;
    var env = try Env.init(std.testing.allocator, content);
    defer env.deinit();

    try std.testing.expect(env.get("no key") == null);
    try std.testing.expectEqualStrings("mysecretpassword", env.get("password").?);
    try std.testing.expectEqual(@as(usize, 123), try env.parseUsize("number"));
    try std.testing.expectEqualStrings("somekey", env.get("somekey").?);
}

test "parse_key returns a value from content" {
    const content =
        \\password=mysecretpassword
        \\number=123
    ;
    try std.testing.expectEqualStrings("mysecretpassword", (try Env.parse_key("password", content)).?);
    try std.testing.expect(try Env.parse_key("missing", content) == null);
}

// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const gguf = @import("gguf");

pub const Allocator = std.mem.Allocator;

pub const Error = error{
    InvalidTokenizer,
    MissingToken,
    UnknownToken,
};

const TokenEntry = struct {
    token: []const u8,
    special: bool,
};

pub const Tokenizer = struct {
    allocator: Allocator,
    token_to_id: std.StringHashMap(u32),
    merge_ranks: std.StringHashMap(u32),
    id_to_token: []?[]u8,
    special_token: []bool,
    eos_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,

    pub fn loadFromDirectory(allocator: Allocator, io: std.Io, model_dir: []const u8) !Tokenizer {
        const path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer.json" });
        defer allocator.free(path);
        const json_bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(json_bytes);
        return fromJson(allocator, json_bytes);
    }

    pub fn loadFromGgufFile(allocator: Allocator, io: std.Io, path: []const u8) !Tokenizer {
        var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        var map = try std.Io.File.MemoryMap.create(io, file, .{
            .len = @intCast(stat.size),
            .protection = .{ .read = true, .write = false },
            .populate = false,
        });
        defer map.destroy(io);

        var parsed = try gguf.parse(allocator, map.memory);
        defer parsed.deinit();
        return fromGguf(allocator, parsed);
    }

    pub fn fromJson(allocator: Allocator, json_bytes: []const u8) !Tokenizer {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{ .parse_numbers = true });
        defer parsed.deinit();
        if (parsed.value != .object) return Error.InvalidTokenizer;

        const model = parsed.value.object.get("model") orelse return Error.InvalidTokenizer;
        if (model != .object) return Error.InvalidTokenizer;
        const vocab = model.object.get("vocab") orelse return Error.InvalidTokenizer;
        if (vocab != .object) return Error.InvalidTokenizer;
        const merges = model.object.get("merges") orelse return Error.InvalidTokenizer;
        if (merges != .array) return Error.InvalidTokenizer;

        var max_id: usize = 0;
        var it = vocab.object.iterator();
        while (it.next()) |entry| {
            max_id = @max(max_id, try jsonUsize(entry.value_ptr.*));
        }

        if (parsed.value.object.get("added_tokens")) |added| {
            if (added == .array) {
                for (added.array.items) |item| {
                    if (item != .object) continue;
                    if (item.object.get("id")) |id_value| {
                        max_id = @max(max_id, try jsonUsize(id_value));
                    }
                }
            }
        }

        var tokenizer = Tokenizer{
            .allocator = allocator,
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .merge_ranks = std.StringHashMap(u32).init(allocator),
            .id_to_token = try allocator.alloc(?[]u8, max_id + 1),
            .special_token = try allocator.alloc(bool, max_id + 1),
        };
        errdefer tokenizer.deinit();
        @memset(tokenizer.id_to_token, null);
        @memset(tokenizer.special_token, false);

        it = vocab.object.iterator();
        while (it.next()) |entry| {
            try tokenizer.addToken(entry.key_ptr.*, @intCast(try jsonUsize(entry.value_ptr.*)), false);
        }

        for (merges.array.items, 0..) |merge, rank| {
            try tokenizer.addMerge(merge, @intCast(rank));
        }

        if (parsed.value.object.get("added_tokens")) |added| {
            if (added == .array) {
                for (added.array.items) |item| {
                    if (item != .object) continue;
                    const content = getString(item.object, "content") orelse continue;
                    const id_value = item.object.get("id") orelse continue;
                    const id: u32 = @intCast(try jsonUsize(id_value));
                    const special = getBool(item.object, "special") orelse false;
                    try tokenizer.addToken(content, id, special);
                    if (std.mem.eql(u8, content, "<|endoftext|>") or std.mem.eql(u8, content, "<|im_end|>")) {
                        tokenizer.eos_token_id = id;
                    }
                }
            }
        }

        return tokenizer;
    }

    pub fn fromGguf(allocator: Allocator, file: gguf.File) !Tokenizer {
        const tokens = try file.stringArrayAlloc(allocator, "tokenizer.ggml.tokens");
        defer allocator.free(tokens);

        const merges = file.stringArrayAlloc(allocator, "tokenizer.ggml.merges") catch |err| switch (err) {
            gguf.Error.InvalidGguf => &.{},
            else => return err,
        };
        defer if (merges.len != 0) allocator.free(merges);

        var tokenizer = Tokenizer{
            .allocator = allocator,
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .merge_ranks = std.StringHashMap(u32).init(allocator),
            .id_to_token = try allocator.alloc(?[]u8, tokens.len),
            .special_token = try allocator.alloc(bool, tokens.len),
            .eos_token_id = if (file.usizeValue("tokenizer.ggml.eos_token_id")) |id| @intCast(id) else null,
            .pad_token_id = if (file.usizeValue("tokenizer.ggml.padding_token_id")) |id| @intCast(id) else null,
        };
        errdefer tokenizer.deinit();
        @memset(tokenizer.id_to_token, null);
        @memset(tokenizer.special_token, false);

        for (tokens, 0..) |token, id| {
            try tokenizer.addToken(token, @intCast(id), looksSpecial(token));
        }
        for (merges, 0..) |merge, rank| {
            try tokenizer.addMergeText(merge, @intCast(rank));
        }

        return tokenizer;
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.id_to_token) |maybe_token| {
            if (maybe_token) |token| self.allocator.free(token);
        }
        var merge_it = self.merge_ranks.iterator();
        while (merge_it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.allocator.free(self.id_to_token);
        self.allocator.free(self.special_token);
        self.merge_ranks.deinit();
        self.token_to_id.deinit();
        self.* = undefined;
    }

    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.token_to_id.get(token);
    }

    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out = std.array_list.Managed(u32).init(allocator);
        errdefer out.deinit();

        var index: usize = 0;
        while (index < text.len) {
            if (try self.matchSpecial(&out, text, &index)) continue;

            const end = self.nextSpecialStart(text, index) orelse text.len;
            try self.encodeNormalChunk(allocator, &out, text[index..end]);
            index = end;
        }

        return out.toOwnedSlice();
    }

    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) ![]u8 {
        var out = std.array_list.Managed(u8).init(allocator);
        errdefer out.deinit();

        for (ids) |id| {
            try self.appendDecodedToken(&out, id);
        }

        return out.toOwnedSlice();
    }

    pub fn decodeToken(self: *const Tokenizer, allocator: Allocator, id: u32) ![]u8 {
        var out = std.array_list.Managed(u8).init(allocator);
        errdefer out.deinit();
        try self.appendDecodedToken(&out, id);
        return out.toOwnedSlice();
    }

    pub fn writeDecodedToken(self: *const Tokenizer, writer: *std.Io.Writer, id: u32) !void {
        const entry = try self.tokenEntry(id);
        if (entry.special) {
            try writer.writeAll(entry.token);
        } else {
            try writeByteLevelDecoded(writer, entry.token);
        }
    }

    fn addToken(self: *Tokenizer, token: []const u8, id: u32, special: bool) !void {
        const id_index: usize = @intCast(id);
        if (id_index >= self.id_to_token.len) return Error.InvalidTokenizer;
        if (self.id_to_token[id_index] != null) {
            self.special_token[id_index] = self.special_token[id_index] or special;
            return;
        }
        const owned = try self.allocator.dupe(u8, token);
        self.id_to_token[id_index] = owned;
        self.special_token[id_index] = special;
        try self.token_to_id.put(owned, id);
    }

    fn appendDecodedToken(self: *const Tokenizer, out: *std.array_list.Managed(u8), id: u32) !void {
        const entry = try self.tokenEntry(id);
        if (entry.special) {
            try out.appendSlice(entry.token);
        } else {
            try appendByteLevelDecoded(out, entry.token);
        }
    }

    fn tokenEntry(self: *const Tokenizer, id: u32) Error!TokenEntry {
        const id_index: usize = @intCast(id);
        if (id_index >= self.id_to_token.len) return Error.MissingToken;
        return .{
            .token = self.id_to_token[id_index] orelse return Error.MissingToken,
            .special = self.special_token[id_index],
        };
    }

    fn addMerge(self: *Tokenizer, merge: std.json.Value, rank: u32) !void {
        var left: []const u8 = undefined;
        var right: []const u8 = undefined;
        switch (merge) {
            .array => |array| {
                if (array.items.len != 2) return Error.InvalidTokenizer;
                left = switch (array.items[0]) {
                    .string => |s| s,
                    else => return Error.InvalidTokenizer,
                };
                right = switch (array.items[1]) {
                    .string => |s| s,
                    else => return Error.InvalidTokenizer,
                };
            },
            .string => |s| {
                try self.addMergeText(s, rank);
                return;
            },
            else => return Error.InvalidTokenizer,
        }
        const key = try pairKey(self.allocator, left, right);
        errdefer self.allocator.free(key);
        try self.merge_ranks.put(key, rank);
    }

    fn addMergeText(self: *Tokenizer, text: []const u8, rank: u32) !void {
        const split = std.mem.indexOfScalar(u8, text, ' ') orelse return Error.InvalidTokenizer;
        const left = text[0..split];
        const right = text[split + 1 ..];
        const key = try pairKey(self.allocator, left, right);
        errdefer self.allocator.free(key);
        try self.merge_ranks.put(key, rank);
    }

    fn encodeNormalChunk(
        self: *const Tokenizer,
        allocator: Allocator,
        out: *std.array_list.Managed(u32),
        chunk: []const u8,
    ) !void {
        if (chunk.len == 0) return;

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const scratch = arena.allocator();

        var symbols = std.array_list.Managed([]const u8).init(scratch);
        for (chunk) |byte| {
            var buf: [4]u8 = undefined;
            const len = try std.unicode.utf8Encode(byteToUnicode(byte), &buf);
            try symbols.append(try scratch.dupe(u8, buf[0..len]));
        }

        while (symbols.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_index: ?usize = null;

            for (0..symbols.items.len - 1) |i| {
                const key = try pairKey(scratch, symbols.items[i], symbols.items[i + 1]);
                if (self.merge_ranks.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_index = i;
                    }
                }
            }

            const merge_index = best_index orelse break;
            const merged = try concat(scratch, symbols.items[merge_index], symbols.items[merge_index + 1]);
            symbols.items[merge_index] = merged;
            _ = symbols.orderedRemove(merge_index + 1);
        }

        for (symbols.items) |symbol| {
            const id = self.token_to_id.get(symbol) orelse return Error.UnknownToken;
            try out.append(id);
        }
    }

    fn matchSpecial(
        self: *const Tokenizer,
        out: *std.array_list.Managed(u32),
        text: []const u8,
        index: *usize,
    ) !bool {
        if (text[index.*] != '<') return false;

        var best_id: ?usize = null;
        var best_len: usize = 0;
        for (self.id_to_token, 0..) |maybe_token, id| {
            if (!self.special_token[id]) continue;
            const token = maybe_token orelse continue;
            if (token.len == 0 or token[0] != '<') continue;
            if (std.mem.startsWith(u8, text[index.*..], token)) {
                if (token.len > best_len) {
                    best_id = id;
                    best_len = token.len;
                }
            }
        }

        if (best_id) |id| {
            try out.append(@intCast(id));
            index.* += best_len;
            return true;
        }
        return false;
    }

    fn nextSpecialStart(self: *const Tokenizer, text: []const u8, start: usize) ?usize {
        var index = start + 1;
        while (index < text.len) : (index += 1) {
            if (text[index] != '<') continue;
            for (self.id_to_token, 0..) |maybe_token, id| {
                if (!self.special_token[id]) continue;
                const token = maybe_token orelse continue;
                if (token.len == 0 or token[0] != '<') continue;
                if (std.mem.startsWith(u8, text[index..], token)) return index;
            }
        }
        return null;
    }
};

fn getString(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

fn getBool(object: std.json.ObjectMap, key: []const u8) ?bool {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .bool => |b| b,
        else => null,
    };
}

fn looksSpecial(token: []const u8) bool {
    return (token.len >= 4 and std.mem.startsWith(u8, token, "<|") and std.mem.endsWith(u8, token, "|>")) or
        std.mem.eql(u8, token, "<s>") or
        std.mem.eql(u8, token, "</s>") or
        std.mem.eql(u8, token, "<unk>");
}

fn jsonUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |i| if (i >= 0) @intCast(i) else Error.InvalidTokenizer,
        .number_string => |s| std.fmt.parseInt(usize, s, 10) catch Error.InvalidTokenizer,
        else => Error.InvalidTokenizer,
    };
}

fn pairKey(allocator: Allocator, left: []const u8, right: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, left.len + 1 + right.len);
    @memcpy(out[0..left.len], left);
    out[left.len] = 0;
    @memcpy(out[left.len + 1 ..], right);
    return out;
}

fn concat(allocator: Allocator, left: []const u8, right: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, left.len + right.len);
    @memcpy(out[0..left.len], left);
    @memcpy(out[left.len..], right);
    return out;
}

fn appendByteLevelDecoded(out: *std.array_list.Managed(u8), token: []const u8) !void {
    var view = try std.unicode.Utf8View.init(token);
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        try out.append(try unicodeToByte(cp));
    }
}

fn writeByteLevelDecoded(writer: *std.Io.Writer, token: []const u8) !void {
    var view = try std.unicode.Utf8View.init(token);
    var it = view.iterator();
    var byte_buf: [1]u8 = undefined;
    while (it.nextCodepoint()) |cp| {
        byte_buf[0] = try unicodeToByte(cp);
        try writer.writeAll(&byte_buf);
    }
}

fn byteToUnicode(byte: u8) u21 {
    if (isByteVisible(byte)) return byte;

    var n: u21 = 0;
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        const current: u8 = @intCast(b);
        if (isByteVisible(current)) continue;
        if (current == byte) return 256 + n;
        n += 1;
    }
    unreachable;
}

fn unicodeToByte(cp: u21) Error!u8 {
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        const byte: u8 = @intCast(b);
        if (byteToUnicode(byte) == cp) return byte;
    }
    return Error.InvalidTokenizer;
}

fn isByteVisible(byte: u8) bool {
    return (byte >= '!' and byte <= '~') or
        (byte >= 0xA1 and byte <= 0xAC) or
        (byte >= 0xAE and byte <= 0xFF);
}

test "loads a minimal tokenizer json and round trips known tokens" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.fromJson(allocator,
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "h": 0,
        \\      "e": 1,
        \\      "l": 2,
        \\      "o": 3,
        \\      "\u0120": 4,
        \\      "w": 5,
        \\      "r": 6,
        \\      "d": 7
        \\    },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [
        \\    {"id": 8, "content": "<|endoftext|>", "special": true}
        \\  ]
        \\}
    );
    defer tok.deinit();

    const ids = try tok.encode(allocator, "hello world<|endoftext|>");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 2, 3, 4, 5, 3, 6, 2, 7, 8 }, ids);

    const text = try tok.decode(allocator, ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("hello world<|endoftext|>", text);

    const one_token = try tok.decodeToken(allocator, 4);
    defer allocator.free(one_token);
    try std.testing.expectEqualStrings(" ", one_token);
}

test "special token matching prefers complete marker boundaries" {
    const allocator = std.testing.allocator;
    var tok = try Tokenizer.fromJson(allocator,
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<": 0,
        \\      "|": 1,
        \\      "x": 2,
        \\      ">": 3
        \\    },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [
        \\    {"id": 4, "content": "<|x|>", "special": true}
        \\  ]
        \\}
    );
    defer tok.deinit();

    const ids = try tok.encode(allocator, "<|x|>");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{4}, ids);
}

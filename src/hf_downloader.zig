// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const Error = error{
    InvalidRepoId,
    InvalidApiResponse,
    DownloadFailed,
    MissingCurl,
};

pub const ModelRef = struct {
    repo_id: []const u8,
    revision: []const u8 = "main",
};

pub const Snapshot = struct {
    allocator: Allocator,
    repo: ModelRef,
    directory: []u8,

    pub fn deinit(self: *Snapshot) void {
        self.allocator.free(self.directory);
        self.* = undefined;
    }
};

pub const RequiredFiles = [_][]const u8{
    "config.json",
    "tokenizer.json",
    "model.safetensors",
};

pub const OptionalFiles = [_][]const u8{
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
};

pub fn validateRepoId(repo_id: []const u8) Error!void {
    if (repo_id.len == 0) return Error.InvalidRepoId;
    var slash_count: usize = 0;
    for (repo_id) |c| {
        switch (c) {
            'a'...'z', 'A'...'Z', '0'...'9', '.', '_', '-', '/' => {},
            else => return Error.InvalidRepoId,
        }
        if (c == '/') slash_count += 1;
    }
    if (slash_count != 1) return Error.InvalidRepoId;
}

pub fn cacheDirName(allocator: Allocator, repo_id: []const u8) ![]u8 {
    try validateRepoId(repo_id);
    const out = try allocator.alloc(u8, repo_id.len);
    for (repo_id, 0..) |c, i| out[i] = if (c == '/') '-' else c;
    return out;
}

pub fn snapshotDir(allocator: Allocator, cache_root: []const u8, repo: ModelRef) ![]u8 {
    const repo_dir = try cacheDirName(allocator, repo.repo_id);
    defer allocator.free(repo_dir);
    return std.fs.path.join(allocator, &.{ cache_root, repo_dir, repo.revision });
}

pub fn resolveUrl(allocator: Allocator, repo: ModelRef, file_path: []const u8) ![]u8 {
    try validateRepoId(repo.repo_id);
    return std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/{s}/resolve/{s}/{s}?download=true",
        .{ repo.repo_id, repo.revision, file_path },
    );
}

pub fn apiUrl(allocator: Allocator, repo: ModelRef) ![]u8 {
    try validateRepoId(repo.repo_id);
    return std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/api/models/{s}/revision/{s}",
        .{ repo.repo_id, repo.revision },
    );
}

pub fn localPath(allocator: Allocator, directory: []const u8, file_path: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ directory, file_path });
}

pub fn ensureSnapshot(
    allocator: Allocator,
    io: std.Io,
    cache_root: []const u8,
    repo: ModelRef,
) !Snapshot {
    return ensureSnapshotForFiles(allocator, io, cache_root, repo, &RequiredFiles, &OptionalFiles);
}

pub fn ensureSnapshotForFiles(
    allocator: Allocator,
    io: std.Io,
    cache_root: []const u8,
    repo: ModelRef,
    required_files: []const []const u8,
    optional_files: []const []const u8,
) !Snapshot {
    try validateRepoId(repo.repo_id);
    const directory = try snapshotDir(allocator, cache_root, repo);
    errdefer allocator.free(directory);
    try std.Io.Dir.cwd().createDirPath(io, directory);

    for (required_files) |file_path| {
        try downloadFile(allocator, io, repo, file_path, directory);
    }
    for (optional_files) |file_path| {
        downloadFile(allocator, io, repo, file_path, directory) catch {};
    }

    return .{ .allocator = allocator, .repo = repo, .directory = directory };
}

pub fn downloadFile(
    allocator: Allocator,
    io: std.Io,
    repo: ModelRef,
    file_path: []const u8,
    directory: []const u8,
) !void {
    const url = try resolveUrl(allocator, repo, file_path);
    defer allocator.free(url);

    const out_path = try localPath(allocator, directory, file_path);
    defer allocator.free(out_path);
    if (try existingFileHasBytes(io, out_path)) return;

    const argv = [_][]const u8{
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--create-dirs",
        "-o",
        out_path,
        url,
    };

    const result = std.process.run(allocator, io, .{
        .argv = &argv,
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(16 * 1024),
    }) catch |err| switch (err) {
        error.FileNotFound => return Error.MissingCurl,
        else => return err,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) return Error.DownloadFailed,
        else => return Error.DownloadFailed,
    }
}

pub fn fetchFileAlloc(
    allocator: Allocator,
    io: std.Io,
    repo: ModelRef,
    file_path: []const u8,
    limit: usize,
) ![]u8 {
    const url = try resolveUrl(allocator, repo, file_path);
    defer allocator.free(url);

    const argv = [_][]const u8{
        "curl",
        "-L",
        "--fail",
        "-s",
        "--retry",
        "3",
        url,
    };

    const result = std.process.run(allocator, io, .{
        .argv = &argv,
        .stdout_limit = .limited(limit),
        .stderr_limit = .limited(16 * 1024),
    }) catch |err| switch (err) {
        error.FileNotFound => return Error.MissingCurl,
        else => return err,
    };
    defer allocator.free(result.stderr);
    errdefer allocator.free(result.stdout);

    switch (result.term) {
        .exited => |code| if (code != 0) return Error.DownloadFailed,
        else => return Error.DownloadFailed,
    }

    return result.stdout;
}

pub fn fetchRepoFileList(allocator: Allocator, io: std.Io, repo: ModelRef) ![][]u8 {
    const url = try apiUrl(allocator, repo);
    defer allocator.free(url);

    const argv = [_][]const u8{
        "curl",
        "-L",
        "--fail",
        "-s",
        "--retry",
        "3",
        url,
    };

    const result = std.process.run(allocator, io, .{
        .argv = &argv,
        .stdout_limit = .limited(4 * 1024 * 1024),
        .stderr_limit = .limited(16 * 1024),
    }) catch |err| switch (err) {
        error.FileNotFound => return Error.MissingCurl,
        else => return err,
    };
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);

    switch (result.term) {
        .exited => |code| if (code != 0) return Error.DownloadFailed,
        else => return Error.DownloadFailed,
    }

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, result.stdout, .{ .parse_numbers = true });
    defer parsed.deinit();
    if (parsed.value != .object) return Error.InvalidApiResponse;
    const siblings = parsed.value.object.get("siblings") orelse return Error.InvalidApiResponse;
    if (siblings != .array) return Error.InvalidApiResponse;

    var out = std.array_list.Managed([]u8).init(allocator);
    errdefer {
        for (out.items) |name| allocator.free(name);
        out.deinit();
    }

    for (siblings.array.items) |item| {
        if (item != .object) continue;
        const name_value = item.object.get("rfilename") orelse continue;
        const name = switch (name_value) {
            .string => |s| s,
            else => continue,
        };
        try out.append(try allocator.dupe(u8, name));
    }

    return out.toOwnedSlice();
}

pub fn freeRepoFileList(allocator: Allocator, files: [][]u8) void {
    for (files) |name| allocator.free(name);
    allocator.free(files);
}

pub fn containsFile(files: []const []const u8, target: []const u8) bool {
    for (files) |file| {
        if (std.mem.eql(u8, file, target)) return true;
    }
    return false;
}

pub fn chooseDefaultNativeGguf(files: []const []const u8) ?[]const u8 {
    const preferred_suffixes = [_][]const u8{
        "BF16.gguf",
        "F16.gguf",
        "F32.gguf",
    };

    for (preferred_suffixes) |suffix| {
        for (files) |file| {
            if (std.mem.endsWith(u8, file, suffix)) return file;
        }
    }
    return null;
}

pub fn hasGguf(files: []const []const u8) bool {
    for (files) |file| {
        if (std.mem.endsWith(u8, file, ".gguf")) return true;
    }
    return false;
}

fn existingFileHasBytes(io: std.Io, path: []const u8) !bool {
    var file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer file.close(io);

    const stat = try file.stat(io);
    return stat.size != 0;
}

test "builds stable hugging face resolve URLs" {
    const allocator = std.testing.allocator;
    const repo: ModelRef = .{ .repo_id = "unsloth/Qwen2.5-0.5B-Instruct" };
    const url = try resolveUrl(allocator, repo, "config.json");
    defer allocator.free(url);

    try std.testing.expectEqualStrings(
        "https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct/resolve/main/config.json?download=true",
        url,
    );
}

test "sanitizes repo id into local cache directory name" {
    const allocator = std.testing.allocator;
    const dir = try cacheDirName(allocator, "org/model");
    defer allocator.free(dir);

    try std.testing.expectEqualStrings("org-model", dir);
    try std.testing.expectError(Error.InvalidRepoId, validateRepoId("bad repo"));
}

test "chooses native gguf before quantized files" {
    const files = [_][]const u8{
        "Model-Q4_K_M.gguf",
        "Model-BF16.gguf",
        "Model-F16.gguf",
    };

    try std.testing.expectEqualStrings("Model-BF16.gguf", chooseDefaultNativeGguf(&files).?);
    try std.testing.expect(hasGguf(&files));
    try std.testing.expect(containsFile(&files, "Model-Q4_K_M.gguf"));
}

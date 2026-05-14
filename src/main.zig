// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const generator = @import("generator");
const hf_downloader = @import("hf_downloader");
const llm = @import("llm");
const tokenizer_mod = @import("tokenizer");

const ModelFormat = enum {
    auto,
    safetensors,
    gguf,
};

const Cli = struct {
    repo_id: []const u8 = "unsloth/Qwen2.5-0.5B-Instruct",
    revision: []const u8 = "main",
    cache_root: []const u8 = ".vvli-cache",
    weights_file: ?[]const u8 = null,
    format: ModelFormat = .auto,
    prompt: ?[]const u8 = null,
    max_new_tokens: usize = 64,
    context: usize = 512,
    threads: usize = 0,
    download: bool = true,
    chat_template: bool = true,

    fn resolveFormat(self: Cli) ModelFormat {
        if (self.format != .auto) return self.format;
        if (self.weights_file) |file| {
            if (std.mem.endsWith(u8, file, ".gguf")) return .gguf;
        }
        if (repoLooksGguf(self.repo_id)) return .gguf;
        return .safetensors;
    }
};

pub fn main(init: std.process.Init) !void {
    run(init) catch |err| {
        try writeAppError(init.io, err);
        return err;
    };
}

fn run(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const allocator = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const cli = parseArgs(args) catch {
        try writeUsage(stdout);
        try stdout.flush();
        return;
    };

    const prompt = cli.prompt orelse {
        try writeUsage(stdout);
        try stdout.flush();
        return;
    };
    const format = cli.resolveFormat();

    const repo: hf_downloader.ModelRef = .{ .repo_id = cli.repo_id, .revision = cli.revision };

    var repo_files: ?[][]u8 = null;
    defer if (repo_files) |files| hf_downloader.freeRepoFileList(allocator, files);

    var owned_weights_file: ?[]u8 = null;
    defer if (owned_weights_file) |file| allocator.free(file);

    const weights_file = try resolveWeightsFile(allocator, io, cli, repo, format, &repo_files, &owned_weights_file);
    const required_files = try resolveRequiredFiles(allocator, io, cli, repo, format, weights_file, &repo_files);
    defer allocator.free(required_files);

    var snapshot = if (cli.download)
        try hf_downloader.ensureSnapshotForFiles(allocator, io, cli.cache_root, repo, required_files, resolveOptionalFiles(format))
    else
        hf_downloader.Snapshot{
            .allocator = allocator,
            .repo = repo,
            .directory = try hf_downloader.snapshotDir(allocator, cli.cache_root, repo),
        };
    defer snapshot.deinit();

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot.directory, weights_file });
    defer allocator.free(weights_path);

    var model = switch (format) {
        .safetensors => safetensors: {
            const config_path = try std.fs.path.join(allocator, &.{ snapshot.directory, "config.json" });
            defer allocator.free(config_path);
            const config = try llm.Config.loadFromFile(allocator, io, config_path);
            break :safetensors try llm.loadOwnedFromSafetensorsFile(allocator, io, config, .bf16, weights_path);
        },
        .gguf => try llm.loadOwnedFromGgufFile(allocator, io, .bf16, weights_path),
        .auto => unreachable,
    };
    defer model.deinit();

    var tokenizer = switch (format) {
        .safetensors => try tokenizer_mod.Tokenizer.loadFromDirectory(allocator, io, snapshot.directory),
        .gguf => try tokenizer_mod.Tokenizer.loadFromGgufFile(allocator, io, weights_path),
        .auto => unreachable,
    };
    defer tokenizer.deinit();

    const rendered_prompt = if (cli.chat_template)
        try renderChatPrompt(allocator, model.config.architecture, prompt)
    else
        try allocator.dupe(u8, prompt);
    defer allocator.free(rendered_prompt);

    const prompt_tokens = try tokenizer.encode(allocator, rendered_prompt);
    defer allocator.free(prompt_tokens);

    const needed_context = prompt_tokens.len + cli.max_new_tokens + 1;
    const requested_context = @max(cli.context, needed_context);
    const context = @min(requested_context, model.config.max_position_embeddings);
    if (needed_context > context) return error.ContextFull;

    var cache = try llm.KVCache.init(allocator, model.config, context);
    defer cache.deinit();
    cache.clear();

    var scratch = try llm.Scratch.init(allocator, model.config, context);
    defer scratch.deinit();

    var runner = try llm.Runner.init(&model.weights, &cache, &scratch, .{
        .thread_count = cli.threads,
        .preload_layers_ahead = 1,
    });

    const generated = try generator.generateGreedyMeasured(allocator, io, &runner, prompt_tokens, .{
        .max_new_tokens = cli.max_new_tokens,
        .eos_token_id = tokenizer.eos_token_id orelse model.config.eos_token_id,
    });
    defer allocator.free(generated.tokens);

    const text = try tokenizer.decode(allocator, generated.tokens);
    defer allocator.free(text);

    try stdout.print(
        "{s}\n\n[{d} output tokens | {d:.2} tok/sec | {d:.2}s decode]\n",
        .{
            text,
            generated.stats.generated_tokens,
            generated.stats.decodeTokensPerSecond(),
            generated.stats.decodeSeconds(),
        },
    );
    try stdout.flush();
}

fn writeAppError(io: std.Io, err: anyerror) !void {
    var stderr_buffer: [2048]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;

    switch (err) {
        error.MissingWeightsFile => try stderr.writeAll("vvli: GGUF requires a weight file when one cannot be selected automatically. Pass --weights <file>. For example: --weights Llama-3.2-1B-Instruct-BF16.gguf\n"),
        error.QuantizedGgufUnsupported => try stderr.writeAll("vvli: this GGUF repo only exposed quantized GGUF files for auto-selection. Quantized GGUF dequant kernels are not implemented yet; pass a BF16/F16/F32 .gguf file if the repo has one.\n"),
        error.MoeRuntimeUnsupported => try stderr.writeAll("vvli: this model is MoE. Config detection is wired, but router/top-k expert execution is not implemented yet, so it is not routed through the dense runner.\n"),
        error.ShardedSafetensorsUnsupported => try stderr.writeAll("vvli: this repo uses sharded safetensors. Shard index parsing and multi-file safetensors loading are not implemented yet.\n"),
        error.UnsupportedDType => try stderr.writeAll("vvli: unsupported tensor dtype. Native BF16/F16/F32 GGUF tensors are supported first; quantized GGUF tensors still need dequant kernels.\n"),
        error.DownloadFailed => try stderr.writeAll("vvli: download failed. Check the repo, revision, selected --format, and --weights file name.\n"),
        else => {},
    }

    try stderr.flush();
}

fn repoLooksGguf(repo_id: []const u8) bool {
    return std.mem.endsWith(u8, repo_id, "-GGUF") or std.mem.endsWith(u8, repo_id, "-gguf");
}

fn resolveWeightsFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    cli: Cli,
    repo: hf_downloader.ModelRef,
    format: ModelFormat,
    repo_files: *?[][]u8,
    owned_weights_file: *?[]u8,
) ![]const u8 {
    if (cli.weights_file) |file| return file;

    switch (format) {
        .safetensors => return "model.safetensors",
        .gguf => {
            if (!cli.download) return error.MissingWeightsFile;
            const files = try getRepoFiles(allocator, io, repo, repo_files);
            const selected = hf_downloader.chooseDefaultNativeGguf(files) orelse {
                if (hf_downloader.hasGguf(files)) return error.QuantizedGgufUnsupported;
                return error.MissingWeightsFile;
            };
            owned_weights_file.* = try allocator.dupe(u8, selected);
            return owned_weights_file.*.?;
        },
        .auto => unreachable,
    }
}

fn resolveRequiredFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    cli: Cli,
    repo: hf_downloader.ModelRef,
    format: ModelFormat,
    weights_file: []const u8,
    repo_files: *?[][]u8,
) ![]const []const u8 {
    switch (format) {
        .gguf => {
            const files = try allocator.alloc([]const u8, 1);
            files[0] = weights_file;
            return files;
        },
        .safetensors => {
            if (cli.download) {
                if (try remoteConfigHasMoeFields(allocator, io, repo)) return error.MoeRuntimeUnsupported;

                const files = try getRepoFiles(allocator, io, repo, repo_files);
                if (!hf_downloader.containsFile(files, weights_file) and hf_downloader.containsFile(files, "model.safetensors.index.json")) {
                    return error.ShardedSafetensorsUnsupported;
                }
            }

            const files = try allocator.alloc([]const u8, 3);
            files[0] = "config.json";
            files[1] = "tokenizer.json";
            files[2] = weights_file;
            return files;
        },
        .auto => unreachable,
    }
}

fn resolveOptionalFiles(format: ModelFormat) []const []const u8 {
    return switch (format) {
        .safetensors => &hf_downloader.OptionalFiles,
        .gguf => &.{},
        .auto => unreachable,
    };
}

fn getRepoFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: hf_downloader.ModelRef,
    repo_files: *?[][]u8,
) ![]const []const u8 {
    if (repo_files.* == null) repo_files.* = try hf_downloader.fetchRepoFileList(allocator, io, repo);
    return repo_files.*.?;
}

fn remoteConfigHasMoeFields(allocator: std.mem.Allocator, io: std.Io, repo: hf_downloader.ModelRef) !bool {
    const json_bytes = try hf_downloader.fetchFileAlloc(allocator, io, repo, "config.json", 1024 * 1024);
    defer allocator.free(json_bytes);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{ .parse_numbers = true });
    defer parsed.deinit();
    if (parsed.value != .object) return false;

    const object = parsed.value.object;
    if (object.get("num_experts") != null) return true;
    if (object.get("num_local_experts") != null) return true;
    if (object.get("num_experts_per_tok") != null) return true;
    if (object.get("moe_intermediate_size") != null) return true;

    if (object.get("model_type")) |value| {
        if (value == .string) {
            const model_type = value.string;
            if (std.mem.indexOf(u8, model_type, "moe") != null or std.mem.indexOf(u8, model_type, "MoE") != null) return true;
        }
    }
    return false;
}

fn parseArgs(args: []const []const u8) !Cli {
    var cli: Cli = .{};
    var i: usize = 1;

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return error.Help;
        if (std.mem.eql(u8, arg, "--repo")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.repo_id = args[i];
        } else if (std.mem.eql(u8, arg, "--revision")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.revision = args[i];
        } else if (std.mem.eql(u8, arg, "--cache")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.cache_root = args[i];
        } else if (std.mem.eql(u8, arg, "--weights")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.weights_file = args[i];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.format = parseFormat(args[i]) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--max-new-tokens")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.max_new_tokens = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--ctx")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.context = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--threads")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.threads = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--no-download")) {
            cli.download = false;
        } else if (std.mem.eql(u8, arg, "--raw")) {
            cli.chat_template = false;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return error.InvalidArgs;
        } else if (cli.prompt == null) {
            cli.prompt = arg;
        } else {
            return error.InvalidArgs;
        }
    }

    return cli;
}

fn parseFormat(text: []const u8) ?ModelFormat {
    if (std.mem.eql(u8, text, "auto")) return .auto;
    if (std.mem.eql(u8, text, "safetensors")) return .safetensors;
    if (std.mem.eql(u8, text, "gguf")) return .gguf;
    return null;
}

fn renderChatPrompt(allocator: std.mem.Allocator, architecture: llm.Architecture, prompt: []const u8) ![]u8 {
    return switch (architecture) {
        .llama => std.fmt.allocPrint(
            allocator,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            .{prompt},
        ),
        .qwen2 => std.fmt.allocPrint(
            allocator,
            "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n",
            .{prompt},
        ),
        .olmoe => std.fmt.allocPrint(allocator, "{s}\n", .{prompt}),
    };
}

fn writeUsage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\usage:
        \\  zig build run -- --repo <owner/model> --prompt "hello"
        \\  zig build run -- --repo unsloth/Qwen2.5-0.5B-Instruct --prompt "Explain CPU inference."
        \\  zig build run -- --repo unsloth/Llama-3.2-1B-Instruct-GGUF --prompt "hello"
        \\
        \\options:
        \\  --repo <id>              Hugging Face repo id. Default: unsloth/Qwen2.5-0.5B-Instruct
        \\  --revision <rev>         Hugging Face revision. Default: main
        \\  --cache <dir>            Local model cache. Default: .vvli-cache
        \\  --format <type>          auto, safetensors, or gguf. Default: auto
        \\  --weights <file>         Weight file in the repo. GGUF auto-selects BF16/F16/F32 when possible
        \\  --prompt <text>          Prompt text
        \\  --max-new-tokens <n>     Default: 64
        \\  --ctx <n>                KV cache length. Default: 512
        \\  --threads <n>            0 uses host CPU count. Default: 0
        \\  --no-download            Use the cache only
        \\  --raw                    Do not wrap prompt in model chat markers
        \\
    );
}

test "renders qwen chat prompt" {
    const allocator = std.testing.allocator;
    const rendered = try renderChatPrompt(allocator, .qwen2, "hi");
    defer allocator.free(rendered);
    try std.testing.expectEqualStrings("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", rendered);
}

test "renders llama chat prompt" {
    const allocator = std.testing.allocator;
    const rendered = try renderChatPrompt(allocator, .llama, "hi");
    defer allocator.free(rendered);
    try std.testing.expectEqualStrings("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", rendered);
}

test "auto format detects gguf repos and files" {
    const cli_by_repo: Cli = .{ .repo_id = "unsloth/Llama-3.2-1B-Instruct-GGUF" };
    try std.testing.expectEqual(ModelFormat.gguf, cli_by_repo.resolveFormat());

    const cli_by_file: Cli = .{ .repo_id = "org/model", .weights_file = "model-BF16.gguf" };
    try std.testing.expectEqual(ModelFormat.gguf, cli_by_file.resolveFormat());

    const cli_default: Cli = .{ .repo_id = "org/model" };
    try std.testing.expectEqual(ModelFormat.safetensors, cli_default.resolveFormat());
}

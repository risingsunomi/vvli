// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const generator = @import("generator");
const hf_downloader = @import("hf_downloader");
const llm = @import("llm");
const tokenizer_mod = @import("tokenizer");
const vision = @import("vision");
const vlm = @import("vlm");

const default_env_path = ".env";

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
    mmproj_file: ?[]const u8 = null,
    format: ModelFormat = .auto,
    prompt: ?[]const u8 = null,
    image_path: ?[]const u8 = null,
    max_new_tokens: usize = 64,
    context: usize = 512,
    threads: usize = 0,
    download: bool = true,
    chat_template: bool = true,
    stream: bool = true,
    temperature: f32 = 0.8,
    top_p: f32 = 0.95,
    top_k: usize = 40,
    repeat_penalty: f32 = 1.10,
    repeat_last_n: usize = 64,
    seed: ?u64 = null,

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
        writeAppError(init.io, err) catch {};
        std.process.exit(1);
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

    if (argsRequestHelp(args)) {
        try writeUsage(stdout);
        try stdout.flush();
        return;
    }

    var cli = try loadCliDefaultsFromDotEnv(arena, io);
    cli = parseArgsWithDefaults(args, cli) catch {
        try writeUsage(stdout);
        try stdout.flush();
        return;
    };

    const prompt = cli.prompt orelse {
        try writeUsage(stdout);
        try stdout.flush();
        return;
    };
    try validateCli(cli);

    if (cli.image_path) |image_path| {
        const image = try vision.inspectPath(io, image_path);
        try runVisionPrompt(allocator, io, cli, prompt, image);
        return;
    }

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

    var progress_context = LoadProgressContext{ .io = io };
    const progress = llm.LoadProgress{
        .context = &progress_context,
        .on_step = reportLoadProgress,
    };

    var model = switch (format) {
        .safetensors => safetensors: {
            const config_path = try std.fs.path.join(allocator, &.{ snapshot.directory, "config.json" });
            defer allocator.free(config_path);
            const config = try llm.Config.loadFromFile(allocator, io, config_path);
            break :safetensors try llm.loadOwnedFromSafetensorsFileWithProgress(allocator, io, config, .bf16, weights_path, progress);
        },
        .gguf => try llm.loadOwnedFromGgufFileWithProgress(allocator, io, .bf16, weights_path, progress),
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

    const generation_options = generator.Options{
        .max_new_tokens = cli.max_new_tokens,
        .eos_token_id = tokenizer.eos_token_id orelse model.config.eos_token_id,
        .temperature = cli.temperature,
        .top_p = cli.top_p,
        .top_k = cli.top_k,
        .repeat_penalty = cli.repeat_penalty,
        .repeat_last_n = cli.repeat_last_n,
        .seed = cli.seed,
    };

    if (cli.stream) {
        var stream_context = StreamContext{
            .tokenizer = &tokenizer,
            .writer = stdout,
            .eos_token_id = generation_options.eos_token_id,
        };
        const generated = try generator.generateGreedyMeasuredStreaming(
            allocator,
            io,
            &runner,
            prompt_tokens,
            generation_options,
            .{
                .context = &stream_context,
                .on_token = streamGeneratedToken,
            },
        );
        defer allocator.free(generated.tokens);

        try stdout.print(
            "\n\n[{d} output tokens | {d:.2} tok/sec | {d:.2}s decode]\n",
            .{
                generated.stats.generated_tokens,
                generated.stats.decodeTokensPerSecond(),
                generated.stats.decodeSeconds(),
            },
        );
    } else {
        const generated = try generator.generateGreedyMeasured(allocator, io, &runner, prompt_tokens, generation_options);
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
    }
    try stdout.flush();
}

fn runVisionPrompt(
    allocator: std.mem.Allocator,
    io: std.Io,
    cli: Cli,
    prompt: []const u8,
    image: vision.ImageInput,
) !void {
    const format = cli.resolveFormat();
    if (format != .gguf) return error.VisionGgufRequired;

    const repo: hf_downloader.ModelRef = .{ .repo_id = cli.repo_id, .revision = cli.revision };

    var repo_files: ?[][]u8 = null;
    defer if (repo_files) |files| hf_downloader.freeRepoFileList(allocator, files);

    var owned_weights_file: ?[]u8 = null;
    defer if (owned_weights_file) |file| allocator.free(file);

    var owned_mmproj_file: ?[]u8 = null;
    defer if (owned_mmproj_file) |file| allocator.free(file);

    const weights_file = try resolveVisionWeightsFile(allocator, io, cli, repo, &repo_files, &owned_weights_file);
    const mmproj_file = try resolveMmprojFile(allocator, io, cli, repo, &repo_files, &owned_mmproj_file);

    const required_files = try allocator.alloc([]const u8, 2);
    defer allocator.free(required_files);
    required_files[0] = weights_file;
    required_files[1] = mmproj_file;

    var snapshot = if (cli.download)
        try hf_downloader.ensureSnapshotForFiles(allocator, io, cli.cache_root, repo, required_files, &.{})
    else
        hf_downloader.Snapshot{
            .allocator = allocator,
            .repo = repo,
            .directory = try hf_downloader.snapshotDir(allocator, cli.cache_root, repo),
        };
    defer snapshot.deinit();

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot.directory, weights_file });
    defer allocator.free(weights_path);

    const mmproj_path = try std.fs.path.join(allocator, &.{ snapshot.directory, mmproj_file });
    defer allocator.free(mmproj_path);

    var plan = try vlm.loadNativePlan(allocator, io, weights_path, mmproj_path, image);
    defer plan.deinit();

    try writeLinkedImage(io, image);
    try writeNativeVisionPlan(io, weights_file, mmproj_file, plan);
    if (plan.projector_quantized_tensors != 0) return error.UnsupportedProjectorDType;

    var decoded = try vlm.decodeResizeForPlan(allocator, plan);
    defer decoded.deinit();

    var patch_embedding = try vlm.loadPatchEmbeddingFromFile(allocator, io, mmproj_path, plan);
    defer patch_embedding.deinit();

    var patch_features = try patch_embedding.forwardWithOptions(allocator, &decoded, .{ .thread_count = cli.threads });
    defer patch_features.deinit();

    try writeNativeVisionStep(io, "running qwen3 vision transformer");
    var transformer = try vlm.loadQwen3VisionTransformerFromFile(allocator, io, mmproj_path, plan);
    defer transformer.deinit();

    var vision_features = try transformer.forward(allocator, &patch_features, .{ .thread_count = cli.threads });
    defer vision_features.deinit();

    var projector = try vlm.loadProjectorFromFile(allocator, io, mmproj_path, plan);
    defer projector.deinit();

    var projected = try projector.forward(allocator, &vision_features, .{ .thread_count = cli.threads });
    defer projected.deinit();

    var tokenizer = try tokenizer_mod.Tokenizer.loadFromGgufFile(allocator, io, weights_path);
    defer tokenizer.deinit();

    var multimodal_prompt = try vlm.buildQwenImagePrompt(allocator, &tokenizer, prompt, projected.tokens);
    defer multimodal_prompt.deinit();

    try writeNativeVisionRun(io, decoded, patch_features, vision_features, projected, multimodal_prompt);
    try writeNativeVisionBoundary(io);
}

const LoadProgressContext = struct {
    io: std.Io,
    last_percent: usize = std.math.maxInt(usize),

    fn report(self: *LoadProgressContext, completed: usize, total: usize, name: []const u8) !void {
        _ = name;
        const percent = if (total == 0) 100 else completed * 100 / total;
        if (completed != total and percent == self.last_percent) return;
        self.last_percent = percent;

        var stderr_buffer: [128]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(self.io, &stderr_buffer);
        try stderr_writer.interface.print("\rloading weights: {d}% ({d}/{d})", .{ percent, completed, total });
        if (completed == total) try stderr_writer.interface.writeAll("\n");
        try stderr_writer.interface.flush();
    }
};

const StreamContext = struct {
    tokenizer: *const tokenizer_mod.Tokenizer,
    writer: *std.Io.Writer,
    eos_token_id: ?u32,
};

fn streamGeneratedToken(context: ?*anyopaque, token: u32, index: usize) !void {
    _ = index;
    const stream_context: *StreamContext = @ptrCast(@alignCast(context.?));
    if (stream_context.eos_token_id) |eos| {
        if (token == eos) return;
    }
    try stream_context.tokenizer.writeDecodedToken(stream_context.writer, token);
    try stream_context.writer.flush();
}

fn reportLoadProgress(context: ?*anyopaque, completed: usize, total: usize, name: []const u8) void {
    if (context) |ptr| {
        const progress_context: *LoadProgressContext = @ptrCast(@alignCast(ptr));
        progress_context.report(completed, total, name) catch {};
    }
}

fn writeLinkedImage(io: std.Io, image: vision.ImageInput) !void {
    var stderr_buffer: [512]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    try stderr_writer.interface.print(
        "linked image: {s} ({d} bytes, {s})\n",
        .{ image.path, image.byte_len, @tagName(image.format) },
    );
    try stderr_writer.interface.flush();
}

fn writeNativeVisionPlan(io: std.Io, weights_file: []const u8, mmproj_file: []const u8, plan: vlm.NativePlan) !void {
    var stderr_buffer: [512]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    try stderr_writer.interface.print(
        "native vision: text={s} ({s}, {d} tensors), mmproj={s} ({s}/{s}, {d}/{d} native tensors), image={d}px patch={d} merge={d}\n",
        .{
            weights_file,
            plan.language_architecture,
            plan.language_tensor_count,
            mmproj_file,
            plan.projector_architecture,
            plan.projector_type,
            plan.projector_native_tensors,
            plan.projector_tensor_count,
            plan.image_size orelse 0,
            plan.patch_size orelse 0,
            plan.spatial_merge_size orelse 0,
        },
    );
    try stderr_writer.interface.flush();
}

fn writeNativeVisionRun(
    io: std.Io,
    decoded: vision.RgbImage,
    patch_features: vlm.ImageEmbeddings,
    vision_features: vlm.ImageEmbeddings,
    projected: vlm.ImageEmbeddings,
    prompt: vlm.MultimodalPrompt,
) !void {
    var stderr_buffer: [768]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    try stderr_writer.interface.print(
        "native vision prepared: decoded={d}x{d} rgb-f32, patches={d}x{d} ({d} tokens x {d}), transformer={d} tokens x {d}, projected={d}x{d} ({d} tokens x {d}), prompt_tokens={d}, image_token_id={d}, image_slots={d}@{d}\n",
        .{
            decoded.width,
            decoded.height,
            patch_features.grid_width,
            patch_features.grid_height,
            patch_features.tokens,
            patch_features.dimensions,
            vision_features.tokens,
            vision_features.dimensions,
            projected.grid_width,
            projected.grid_height,
            projected.tokens,
            projected.dimensions,
            prompt.tokens.len,
            prompt.image_token_id,
            prompt.image_token_count,
            prompt.image_start,
        },
    );
    try stderr_writer.interface.flush();
}

fn writeNativeVisionStep(io: std.Io, message: []const u8) !void {
    var stderr_buffer: [256]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    try stderr_writer.interface.print("native vision: {s}\n", .{message});
    try stderr_writer.interface.flush();
}

fn writeNativeVisionBoundary(io: std.Io) !void {
    var stderr_buffer: [512]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    try stderr_writer.interface.writeAll(
        "native vision status: vision transformer, M-RoPE, projector, and multimodal token insertion completed. Generation is not started yet because the Qwen3.5 text runtime still needs qwen35/SSM and quantized GGUF matmul support.\n",
    );
    try stderr_writer.interface.flush();
}

fn writeAppError(io: std.Io, err: anyerror) !void {
    var stderr_buffer: [2048]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_writer.interface;

    switch (err) {
        error.MissingWeightsFile => try stderr.writeAll("vvli: GGUF requires a weight file when one cannot be selected automatically. Pass --weights <file>. For example: --weights Llama-3.2-1B-Instruct-BF16.gguf\n"),
        error.QuantizedGgufUnsupported => try stderr.writeAll("vvli: this GGUF repo only exposed quantized GGUF files for auto-selection. Quantized GGUF dequant kernels are not implemented yet; pass a BF16/F16/F32 .gguf file if the repo has one.\n"),
        error.MoeRuntimeUnsupported => try stderr.writeAll("vvli: this model is MoE. Config detection is wired, but router/top-k expert execution is not implemented yet, so it is not routed through the dense runner.\n"),
        error.VisionGgufRequired => try stderr.writeAll("vvli: --image currently requires a GGUF vision-language repo. Pass something like --repo unsloth/Qwen3.5-9B-GGUF, or use --format gguf with --weights and --mmproj.\n"),
        error.MissingMmprojFile => try stderr.writeAll("vvli: could not select an mmproj GGUF projector from this repo. Pass --mmproj <file> or use a VLM GGUF repo that contains mmproj-F16.gguf/mmproj-BF16.gguf.\n"),
        error.NativeVisionExecutionIncomplete => try stderr.writeAll("vvli: native image decode/resize, Qwen3VL vision transformer/M-RoPE, projector MLP, and multimodal token insertion are wired. Image-aware Qwen3.5 generation still needs qwen35/SSM text runtime and quantized GGUF matmul support.\n"),
        error.MissingProjectorTensor => try stderr.writeAll("vvli: mmproj is missing expected projector tensors such as mm.0.weight/mm.2.weight for the native VLM path.\n"),
        error.MissingImagePadToken => try stderr.writeAll("vvli: tokenizer does not expose the expected <|image_pad|> special token for multimodal insertion.\n"),
        error.ShardedSafetensorsUnsupported => try stderr.writeAll("vvli: this repo uses sharded safetensors. Shard index parsing and multi-file safetensors loading are not implemented yet.\n"),
        error.UnsupportedImageFormat => try stderr.writeAll("vvli: unsupported image format. Pass a .jpg, .jpeg, .png, .webp, or .bmp path.\n"),
        error.InvalidImageFile => try stderr.writeAll("vvli: image file extension and file header do not match a supported image format.\n"),
        error.NativeImageDecodeUnsupported => try stderr.writeAll("vvli: native image decode/resize currently uses ImageIO on macOS. Non-Apple image decoders are planned.\n"),
        error.EmptyImageFile => try stderr.writeAll("vvli: image file is empty.\n"),
        error.InvalidEnv => try stderr.writeAll("vvli: invalid .env. Expected KEY=VALUE lines with VVLI_* keys; see .env.example.\n"),
        error.UnsupportedDType => try stderr.writeAll("vvli: unsupported tensor dtype. Native BF16/F16/F32 GGUF tensors are supported first; quantized GGUF tensors still need dequant kernels.\n"),
        error.UnsupportedProjectorDType => try stderr.writeAll("vvli: unsupported projector dtype. The native VLM path currently requires BF16/F16/F32 mmproj tensors.\n"),
        error.UnsupportedModelArchitecture => try stderr.writeAll("vvli: this model architecture is not implemented in the CPU runner yet.\n"),
        error.InvalidSamplingOptions => try stderr.writeAll("vvli: invalid sampling options. Use temperature >= 0, 0 < top-p <= 1, and repeat-penalty >= 1.\n"),
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

fn resolveVisionWeightsFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    cli: Cli,
    repo: hf_downloader.ModelRef,
    repo_files: *?[][]u8,
    owned_weights_file: *?[]u8,
) ![]const u8 {
    if (cli.weights_file) |file| return file;
    if (!cli.download) return error.MissingWeightsFile;

    const files = try getRepoFiles(allocator, io, repo, repo_files);
    const selected = hf_downloader.chooseDefaultNativeGguf(files) orelse {
        if (hf_downloader.hasGguf(files)) return error.QuantizedGgufUnsupported;
        return error.MissingWeightsFile;
    };
    owned_weights_file.* = try allocator.dupe(u8, selected);
    return owned_weights_file.*.?;
}

fn resolveMmprojFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    cli: Cli,
    repo: hf_downloader.ModelRef,
    repo_files: *?[][]u8,
    owned_mmproj_file: *?[]u8,
) ![]const u8 {
    if (cli.mmproj_file) |file| return file;
    if (!cli.download) return error.MissingMmprojFile;

    const files = try getRepoFiles(allocator, io, repo, repo_files);
    const selected = hf_downloader.chooseDefaultMmprojGguf(files) orelse return error.MissingMmprojFile;
    owned_mmproj_file.* = try allocator.dupe(u8, selected);
    return owned_mmproj_file.*.?;
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

fn loadCliDefaultsFromDotEnv(allocator: std.mem.Allocator, io: std.Io) !Cli {
    var cli: Cli = .{};
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, default_env_path, allocator, .limited(64 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return cli,
        else => return err,
    };
    try applyDotEnvDefaults(&cli, bytes);
    return cli;
}

fn applyDotEnvDefaults(cli: *Cli, bytes: []const u8) !void {
    var rest = bytes;
    while (rest.len > 0) {
        const newline = std.mem.indexOfScalar(u8, rest, '\n') orelse rest.len;
        const raw_line = rest[0..newline];
        rest = if (newline == rest.len) rest[rest.len..] else rest[newline + 1 ..];

        var line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "export ")) {
            line = std.mem.trim(u8, line["export ".len..], " \t\r\n");
        }

        const separator = std.mem.indexOfScalar(u8, line, '=') orelse return error.InvalidEnv;
        const key = std.mem.trim(u8, line[0..separator], " \t");
        const value = try parseDotEnvValue(line[separator + 1 ..]);
        if (key.len == 0) return error.InvalidEnv;
        try applyDotEnvDefault(cli, key, value);
    }
}

fn parseDotEnvValue(raw: []const u8) ![]const u8 {
    const value = trimDotEnvLeft(raw);
    if (value.len == 0) return "";

    if (value[0] == '"' or value[0] == '\'') {
        const quote = value[0];
        var escaped = false;
        var i: usize = 1;
        while (i < value.len) : (i += 1) {
            const byte = value[i];
            if (quote == '"' and byte == '\\' and !escaped) {
                escaped = true;
                continue;
            }
            if (byte == quote and !escaped) {
                const trailing = std.mem.trim(u8, value[i + 1 ..], " \t\r\n");
                if (trailing.len != 0 and trailing[0] != '#') return error.InvalidEnv;
                return value[1..i];
            }
            escaped = false;
        }
        return error.InvalidEnv;
    }

    return std.mem.trim(u8, stripDotEnvComment(value), " \t\r\n");
}

fn trimDotEnvLeft(value: []const u8) []const u8 {
    var start: usize = 0;
    while (start < value.len and (value[start] == ' ' or value[start] == '\t')) : (start += 1) {}
    return value[start..];
}

fn stripDotEnvComment(value: []const u8) []const u8 {
    var i: usize = 0;
    while (i < value.len) : (i += 1) {
        if (value[i] == '#' and (i == 0 or isDotEnvSpace(value[i - 1]))) return value[0..i];
    }
    return value;
}

fn isDotEnvSpace(byte: u8) bool {
    return byte == ' ' or byte == '\t' or byte == '\r' or byte == '\n';
}

fn applyDotEnvDefault(cli: *Cli, key: []const u8, value: []const u8) !void {
    if (value.len == 0) return;

    if (std.mem.eql(u8, key, "VVLI_REPO")) {
        cli.repo_id = value;
    } else if (std.mem.eql(u8, key, "VVLI_REVISION")) {
        cli.revision = value;
    } else if (std.mem.eql(u8, key, "VVLI_CACHE")) {
        cli.cache_root = value;
    } else if (std.mem.eql(u8, key, "VVLI_WEIGHTS")) {
        cli.weights_file = value;
    } else if (std.mem.eql(u8, key, "VVLI_MMPROJ")) {
        cli.mmproj_file = value;
    } else if (std.mem.eql(u8, key, "VVLI_FORMAT")) {
        cli.format = parseFormat(value) orelse return error.InvalidEnv;
    } else if (std.mem.eql(u8, key, "VVLI_PROMPT")) {
        cli.prompt = value;
    } else if (std.mem.eql(u8, key, "VVLI_IMAGE")) {
        cli.image_path = value;
    } else if (std.mem.eql(u8, key, "VVLI_MAX_NEW_TOKENS")) {
        cli.max_new_tokens = try parseEnvInt(usize, value);
    } else if (std.mem.eql(u8, key, "VVLI_CTX")) {
        cli.context = try parseEnvInt(usize, value);
    } else if (std.mem.eql(u8, key, "VVLI_THREADS")) {
        cli.threads = try parseEnvInt(usize, value);
    } else if (std.mem.eql(u8, key, "VVLI_DOWNLOAD")) {
        cli.download = try parseEnvBool(value);
    } else if (std.mem.eql(u8, key, "VVLI_CHAT_TEMPLATE")) {
        cli.chat_template = try parseEnvBool(value);
    } else if (std.mem.eql(u8, key, "VVLI_STREAM")) {
        cli.stream = try parseEnvBool(value);
    } else if (std.mem.eql(u8, key, "VVLI_TEMPERATURE")) {
        cli.temperature = try parseEnvFloat(f32, value);
    } else if (std.mem.eql(u8, key, "VVLI_TOP_P")) {
        cli.top_p = try parseEnvFloat(f32, value);
    } else if (std.mem.eql(u8, key, "VVLI_TOP_K")) {
        cli.top_k = try parseEnvInt(usize, value);
    } else if (std.mem.eql(u8, key, "VVLI_REPEAT_PENALTY")) {
        cli.repeat_penalty = try parseEnvFloat(f32, value);
    } else if (std.mem.eql(u8, key, "VVLI_REPEAT_LAST_N")) {
        cli.repeat_last_n = try parseEnvInt(usize, value);
    } else if (std.mem.eql(u8, key, "VVLI_SEED")) {
        cli.seed = try parseEnvInt(u64, value);
    } else if (std.mem.eql(u8, key, "VVLI_GREEDY")) {
        if (try parseEnvBool(value)) {
            cli.temperature = 0.0;
            cli.top_p = 1.0;
            cli.top_k = 1;
            cli.repeat_penalty = 1.0;
        }
    }
}

fn parseEnvInt(comptime T: type, value: []const u8) !T {
    return std.fmt.parseInt(T, value, 10) catch return error.InvalidEnv;
}

fn parseEnvFloat(comptime T: type, value: []const u8) !T {
    return std.fmt.parseFloat(T, value) catch return error.InvalidEnv;
}

fn parseEnvBool(value: []const u8) !bool {
    if (std.ascii.eqlIgnoreCase(value, "true") or
        std.mem.eql(u8, value, "1") or
        std.ascii.eqlIgnoreCase(value, "yes") or
        std.ascii.eqlIgnoreCase(value, "on"))
    {
        return true;
    }
    if (std.ascii.eqlIgnoreCase(value, "false") or
        std.mem.eql(u8, value, "0") or
        std.ascii.eqlIgnoreCase(value, "no") or
        std.ascii.eqlIgnoreCase(value, "off"))
    {
        return false;
    }
    return error.InvalidEnv;
}

fn parseArgs(args: []const []const u8) !Cli {
    return parseArgsWithDefaults(args, .{});
}

fn argsRequestHelp(args: []const []const u8) bool {
    if (args.len <= 1) return false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return true;
    }
    return false;
}

fn parseArgsWithDefaults(args: []const []const u8, defaults: Cli) !Cli {
    var cli = defaults;
    var i: usize = 1;
    var prompt_set_by_args = false;

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
        } else if (std.mem.eql(u8, arg, "--mmproj")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.mmproj_file = args[i];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.format = parseFormat(args[i]) orelse return error.InvalidArgs;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.prompt = args[i];
            prompt_set_by_args = true;
        } else if (std.mem.eql(u8, arg, "--image")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.image_path = args[i];
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
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.temperature = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.top_p = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.top_k = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--repeat-penalty")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.repeat_penalty = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--repeat-last-n")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.repeat_last_n = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgs;
            cli.seed = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--greedy")) {
            cli.temperature = 0.0;
            cli.top_p = 1.0;
            cli.top_k = 1;
            cli.repeat_penalty = 1.0;
        } else if (std.mem.eql(u8, arg, "--download")) {
            cli.download = true;
        } else if (std.mem.eql(u8, arg, "--no-download")) {
            cli.download = false;
        } else if (std.mem.eql(u8, arg, "--stream")) {
            cli.stream = true;
        } else if (std.mem.eql(u8, arg, "--no-stream")) {
            cli.stream = false;
        } else if (std.mem.eql(u8, arg, "--chat-template")) {
            cli.chat_template = true;
        } else if (std.mem.eql(u8, arg, "--raw")) {
            cli.chat_template = false;
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return error.InvalidArgs;
        } else if (!prompt_set_by_args) {
            cli.prompt = arg;
            prompt_set_by_args = true;
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

fn validateCli(cli: Cli) !void {
    if (cli.temperature < 0.0 or std.math.isNan(cli.temperature)) return error.InvalidSamplingOptions;
    if (cli.top_p <= 0.0 or cli.top_p > 1.0 or std.math.isNan(cli.top_p)) return error.InvalidSamplingOptions;
    if (cli.repeat_penalty < 1.0 or std.math.isNan(cli.repeat_penalty)) return error.InvalidSamplingOptions;
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
        \\  zig build run -- --repo unsloth/Qwen3.5-9B-GGUF --weights Qwen3.5-9B-Q4_0.gguf --mmproj mmproj-F16.gguf --image ./image.jpg --prompt "Describe this image."
        \\
        \\vvli also reads default values from .env when present. Command-line flags override .env.
        \\
        \\options:
        \\  --repo <id>              Hugging Face repo id. Default: unsloth/Qwen2.5-0.5B-Instruct
        \\  --revision <rev>         Hugging Face revision. Default: main
        \\  --cache <dir>            Local model cache. Default: .vvli-cache
        \\  --format <type>          auto, safetensors, or gguf. Default: auto
        \\  --weights <file>         Weight file in the repo. GGUF auto-selects BF16/F16/F32 when possible
        \\  --mmproj <file>          Multimodal projector file for --image GGUF runs
        \\  --prompt <text>          Prompt text
        \\  --image <path>           Run the native vision-language path
        \\  --max-new-tokens <n>     Default: 64
        \\  --ctx <n>                KV cache length. Default: 512
        \\  --threads <n>            0 uses host CPU count. Default: 0
        \\  --temperature <f>        Sampling temperature. 0 forces greedy. Default: 0.8
        \\  --top-p <f>              Nucleus sampling threshold. Default: 0.95
        \\  --top-k <n>              Candidate cap before top-p. 0 disables. Default: 40
        \\  --repeat-penalty <f>     Penalize recent token repeats. 1 disables. Default: 1.10
        \\  --repeat-last-n <n>      Recent prompt/output window for repeat penalty. Default: 64
        \\  --seed <n>               Fixed sampling seed
        \\  --greedy                 Shortcut for temperature 0, top-p 1, top-k 1, repeat penalty 1
        \\  --download               Download missing files. Default
        \\  --no-download            Use the cache only
        \\  --stream                 Stream generated text as tokens are decoded. Default
        \\  --no-stream              Decode and print only after generation completes
        \\  --chat-template          Wrap prompt in model chat markers. Default
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

test "parses image path option" {
    const cli = try parseArgs(&.{ "vvli", "--prompt", "describe", "--image", "frame.PNG", "--mmproj", "mmproj-F16.gguf" });
    try std.testing.expectEqualStrings("frame.PNG", cli.image_path.?);
    try std.testing.expectEqualStrings("mmproj-F16.gguf", cli.mmproj_file.?);
}

test "streaming is enabled by default and can be disabled" {
    const default_cli = try parseArgs(&.{ "vvli", "--prompt", "hello" });
    try std.testing.expect(default_cli.stream);

    const disabled = try parseArgs(&.{ "vvli", "--prompt", "hello", "--no-stream" });
    try std.testing.expect(!disabled.stream);
}

test "parses sampling options" {
    const cli = try parseArgs(&.{
        "vvli",
        "--prompt",
        "hello",
        "--temperature",
        "0.7",
        "--top-p",
        "0.9",
        "--top-k",
        "20",
        "--repeat-penalty",
        "1.2",
        "--repeat-last-n",
        "32",
        "--seed",
        "1234",
    });
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), cli.temperature, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), cli.top_p, 1e-6);
    try std.testing.expectEqual(@as(usize, 20), cli.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), cli.repeat_penalty, 1e-6);
    try std.testing.expectEqual(@as(usize, 32), cli.repeat_last_n);
    try std.testing.expectEqual(@as(u64, 1234), cli.seed.?);

    const greedy = try parseArgs(&.{ "vvli", "--prompt", "hello", "--greedy" });
    try std.testing.expectEqual(@as(f32, 0.0), greedy.temperature);
    try std.testing.expectEqual(@as(f32, 1.0), greedy.top_p);
    try std.testing.expectEqual(@as(usize, 1), greedy.top_k);
    try std.testing.expectEqual(@as(f32, 1.0), greedy.repeat_penalty);
}

test "parses dotenv defaults" {
    var cli: Cli = .{};
    try applyDotEnvDefaults(&cli,
        \\# comments and blank lines are ignored
        \\VVLI_REPO=org/model
        \\VVLI_REVISION="feature branch"
        \\VVLI_CACHE=.cache/vvli
        \\VVLI_FORMAT=gguf
        \\VVLI_WEIGHTS=model-BF16.gguf
        \\VVLI_MMPROJ='mmproj-F16.gguf'
        \\VVLI_PROMPT=hello from dotenv # trailing comment
        \\VVLI_IMAGE=frame.png
        \\VVLI_MAX_NEW_TOKENS=12
        \\VVLI_CTX=256
        \\VVLI_THREADS=2
        \\VVLI_DOWNLOAD=false
        \\VVLI_CHAT_TEMPLATE=false
        \\VVLI_STREAM=no
        \\VVLI_TEMPERATURE=0.4
        \\VVLI_TOP_P=0.8
        \\VVLI_TOP_K=7
        \\VVLI_REPEAT_PENALTY=1.2
        \\VVLI_REPEAT_LAST_N=16
        \\VVLI_SEED=42
        \\IGNORED_KEY=ignored
    );

    try std.testing.expectEqualStrings("org/model", cli.repo_id);
    try std.testing.expectEqualStrings("feature branch", cli.revision);
    try std.testing.expectEqualStrings(".cache/vvli", cli.cache_root);
    try std.testing.expectEqual(ModelFormat.gguf, cli.format);
    try std.testing.expectEqualStrings("model-BF16.gguf", cli.weights_file.?);
    try std.testing.expectEqualStrings("mmproj-F16.gguf", cli.mmproj_file.?);
    try std.testing.expectEqualStrings("hello from dotenv", cli.prompt.?);
    try std.testing.expectEqualStrings("frame.png", cli.image_path.?);
    try std.testing.expectEqual(@as(usize, 12), cli.max_new_tokens);
    try std.testing.expectEqual(@as(usize, 256), cli.context);
    try std.testing.expectEqual(@as(usize, 2), cli.threads);
    try std.testing.expect(!cli.download);
    try std.testing.expect(!cli.chat_template);
    try std.testing.expect(!cli.stream);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), cli.temperature, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), cli.top_p, 1e-6);
    try std.testing.expectEqual(@as(usize, 7), cli.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), cli.repeat_penalty, 1e-6);
    try std.testing.expectEqual(@as(usize, 16), cli.repeat_last_n);
    try std.testing.expectEqual(@as(u64, 42), cli.seed.?);
}

test "cli args override dotenv defaults" {
    var defaults: Cli = .{};
    try applyDotEnvDefaults(&defaults,
        \\VVLI_REPO=org/env-model
        \\VVLI_PROMPT=env prompt
        \\VVLI_TOP_K=5
        \\VVLI_STREAM=false
        \\VVLI_DOWNLOAD=false
        \\VVLI_CHAT_TEMPLATE=false
    );

    const cli = try parseArgsWithDefaults(&.{
        "vvli",
        "--repo",
        "org/cli-model",
        "--prompt",
        "cli prompt",
        "--top-k",
        "11",
        "--stream",
        "--download",
        "--chat-template",
    }, defaults);

    try std.testing.expectEqualStrings("org/cli-model", cli.repo_id);
    try std.testing.expectEqualStrings("cli prompt", cli.prompt.?);
    try std.testing.expectEqual(@as(usize, 11), cli.top_k);
    try std.testing.expect(cli.stream);
    try std.testing.expect(cli.download);
    try std.testing.expect(cli.chat_template);

    const positional_prompt = try parseArgsWithDefaults(&.{ "vvli", "positional prompt" }, defaults);
    try std.testing.expectEqualStrings("positional prompt", positional_prompt.prompt.?);
}

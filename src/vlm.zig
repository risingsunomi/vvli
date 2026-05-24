// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const gguf = @import("gguf");
const tensor = @import("tensor");
const tokenizer_mod = @import("tokenizer");
const vision = @import("vision");

pub const Allocator = std.mem.Allocator;
const simd_lanes = tensor.simd_lanes;
const Vec = @Vector(simd_lanes, f32);

pub const Error = error{
    InvalidProjectorTensor,
    MissingImagePadToken,
    MissingPatchEmbedding,
    MissingProjectorTensor,
    MissingVisionTensor,
    NativeVisionExecutionIncomplete,
    UnsupportedProjectorArchitecture,
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
    spatial_merge_size: ?usize,
    layer_norm_epsilon: ?f32,
    projector_tensor_count: usize,
    projector_native_tensors: usize,
    projector_quantized_tensors: usize,
    image_mean: [3]f32,
    image_std: [3]f32,

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
        .spatial_merge_size = projector_gguf.usizeValue("clip.vision.spatial_merge_size"),
        .layer_norm_epsilon = projector_gguf.f32Value("clip.vision.attention.layer_norm_epsilon"),
        .projector_tensor_count = projector_tensor_count,
        .projector_native_tensors = projector_native_tensors,
        .projector_quantized_tensors = projector_tensor_count - projector_native_tensors,
        .image_mean = try imageArray3OrDefault(allocator, projector_gguf, "clip.vision.image_mean", .{ 0.48145466, 0.4578275, 0.40821073 }),
        .image_std = try imageArray3OrDefault(allocator, projector_gguf, "clip.vision.image_std", .{ 0.26862954, 0.26130258, 0.27577711 }),
    };
}

pub const ImageEmbeddings = struct {
    allocator: Allocator,
    tokens: usize,
    dimensions: usize,
    grid_width: usize,
    grid_height: usize,
    data: []align(tensor.data_alignment_bytes) f32,

    pub fn deinit(self: *ImageEmbeddings) void {
        if (self.data.len != 0) self.allocator.free(self.data);
        self.* = .{
            .allocator = self.allocator,
            .tokens = 0,
            .dimensions = 0,
            .grid_width = 0,
            .grid_height = 0,
            .data = emptyAlignedF32(),
        };
    }
};

pub const PatchEmbedding = struct {
    allocator: Allocator,
    weight: tensor.Tensor,
    second_weight: ?tensor.Tensor,
    bias: ?tensor.Tensor,
    layout: PatchWeightLayout,
    patch_size: usize,
    in_channels: usize,
    out_channels: usize,

    pub fn deinit(self: *PatchEmbedding) void {
        self.weight.deinit(self.allocator);
        if (self.second_weight) |*second_weight| second_weight.deinit(self.allocator);
        if (self.bias) |*bias| bias.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn forward(self: *const PatchEmbedding, allocator: Allocator, image: *const vision.RgbImage) !ImageEmbeddings {
        return self.forwardWithOptions(allocator, image, .{});
    }

    pub fn forwardWithOptions(
        self: *const PatchEmbedding,
        allocator: Allocator,
        image: *const vision.RgbImage,
        options: ProjectorOptions,
    ) !ImageEmbeddings {
        if (image.width < self.patch_size or image.height < self.patch_size) return Error.InvalidProjectorTensor;
        const out_w = image.width / self.patch_size;
        const out_h = image.height / self.patch_size;
        const tokens = out_w * out_h;
        const data = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * self.out_channels);
        errdefer allocator.free(data);

        try self.forwardRangeThreaded(image, data, out_w, tokens, options);

        return .{
            .allocator = allocator,
            .tokens = tokens,
            .dimensions = self.out_channels,
            .grid_width = out_w,
            .grid_height = out_h,
            .data = data,
        };
    }

    fn forwardRangeThreaded(
        self: *const PatchEmbedding,
        image: *const vision.RgbImage,
        data: []f32,
        out_w: usize,
        tokens: usize,
        options: ProjectorOptions,
    ) !void {
        const workers = resolveProjectorWorkers(tokens, options);
        if (workers == 1) {
            self.forwardTokenRange(image, data, out_w, 0, tokens);
            return;
        }

        var threads: [32]std.Thread = undefined;
        var jobs: [32]PatchEmbeddingJob = undefined;
        const tokens_per_worker = (tokens + workers - 1) / workers;

        var spawned: usize = 0;
        while (spawned + 1 < workers) : (spawned += 1) {
            const start = spawned * tokens_per_worker;
            const end = @min(start + tokens_per_worker, tokens);
            jobs[spawned] = .{
                .patch = self,
                .image = image,
                .data = data,
                .out_w = out_w,
                .start_token = start,
                .end_token = end,
            };
            threads[spawned] = try std.Thread.spawn(.{}, patchEmbeddingWorker, .{&jobs[spawned]});
        }

        const start = spawned * tokens_per_worker;
        self.forwardTokenRange(image, data, out_w, start, tokens);
        for (threads[0..spawned]) |thread| thread.join();
    }

    fn forwardTokenRange(
        self: *const PatchEmbedding,
        image: *const vision.RgbImage,
        data: []f32,
        out_w: usize,
        start_token: usize,
        end_token: usize,
    ) void {
        for (start_token..end_token) |token_index| {
            const py = token_index / out_w;
            const px = token_index % out_w;
            for (0..self.out_channels) |oc| {
                var acc: f32 = if (self.bias) |bias| bias.data[oc] else 0.0;
                for (0..self.patch_size) |ky| {
                    for (0..self.patch_size) |kx| {
                        const x = px * self.patch_size + kx;
                        const y = py * self.patch_size + ky;
                        const pixel_base = (y * image.width + x) * 3;
                        for (0..self.in_channels) |ch| {
                            acc = @mulAdd(f32, image.data[pixel_base + ch], self.weightAt(oc, ch, ky, kx), acc);
                            if (self.second_weight) |second_weight| {
                                acc = @mulAdd(f32, image.data[pixel_base + ch], self.weightAtTensor(second_weight, oc, ch, ky, kx), acc);
                            }
                        }
                    }
                }
                data[token_index * self.out_channels + oc] = acc;
            }
        }
    }

    fn weightAt(self: *const PatchEmbedding, oc: usize, ch: usize, ky: usize, kx: usize) f32 {
        return self.weightAtTensor(self.weight, oc, ch, ky, kx);
    }

    fn weightAtTensor(self: *const PatchEmbedding, weight: tensor.Tensor, oc: usize, ch: usize, ky: usize, kx: usize) f32 {
        const p = self.patch_size;
        return switch (self.layout) {
            .ggml_hwio => weight.data[kx + p * (ky + p * (ch + self.in_channels * oc))],
            .ggml_oihw => weight.data[oc + self.out_channels * (ch + self.in_channels * (ky + p * kx))],
        };
    }
};

pub const ProjectorOptions = struct {
    thread_count: usize = 0,
};

pub const VisionLayer = struct {
    ln1_weight: tensor.Tensor,
    ln1_bias: tensor.Tensor,
    qkv_weight: tensor.Tensor,
    qkv_bias: tensor.Tensor,
    attn_out_weight: tensor.Tensor,
    attn_out_bias: tensor.Tensor,
    ln2_weight: tensor.Tensor,
    ln2_bias: tensor.Tensor,
    ffn_up_weight: tensor.Tensor,
    ffn_up_bias: tensor.Tensor,
    ffn_down_weight: tensor.Tensor,
    ffn_down_bias: tensor.Tensor,

    fn deinit(self: *VisionLayer, allocator: Allocator) void {
        self.ln1_weight.deinit(allocator);
        self.ln1_bias.deinit(allocator);
        self.qkv_weight.deinit(allocator);
        self.qkv_bias.deinit(allocator);
        self.attn_out_weight.deinit(allocator);
        self.attn_out_bias.deinit(allocator);
        self.ln2_weight.deinit(allocator);
        self.ln2_bias.deinit(allocator);
        self.ffn_up_weight.deinit(allocator);
        self.ffn_up_bias.deinit(allocator);
        self.ffn_down_weight.deinit(allocator);
        self.ffn_down_bias.deinit(allocator);
        self.* = undefined;
    }
};

pub const Qwen3VisionTransformer = struct {
    allocator: Allocator,
    position_embeddings: tensor.Tensor,
    layers: []VisionLayer,
    hidden_size: usize,
    intermediate_size: usize,
    head_count: usize,
    epsilon: f32,

    pub fn deinit(self: *Qwen3VisionTransformer) void {
        self.position_embeddings.deinit(self.allocator);
        for (self.layers) |*layer| layer.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    pub fn forward(
        self: *const Qwen3VisionTransformer,
        allocator: Allocator,
        patches: *const ImageEmbeddings,
        options: ProjectorOptions,
    ) !ImageEmbeddings {
        if (patches.dimensions != self.hidden_size) return Error.InvalidProjectorTensor;
        if (patches.tokens != matrixRows(self.position_embeddings)) return Error.InvalidProjectorTensor;
        if (matrixCols(self.position_embeddings) != self.hidden_size) return Error.InvalidProjectorTensor;

        const tokens = patches.tokens;
        const hidden = self.hidden_size;
        const qkv_dim = hidden * 3;
        const head_dim = hidden / self.head_count;
        if (hidden % self.head_count != 0 or head_dim % 2 != 0) return Error.InvalidProjectorTensor;

        const current = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * hidden);
        errdefer allocator.free(current);
        @memcpy(current, patches.data);
        addPositionEmbeddings(current, self.position_embeddings);

        const norm = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * hidden);
        defer allocator.free(norm);
        const qkv = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * qkv_dim);
        defer allocator.free(qkv);
        const attention = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * hidden);
        defer allocator.free(attention);
        const projected = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * hidden);
        defer allocator.free(projected);
        const ffn = try allocator.alignedAlloc(f32, tensor.data_alignment, tokens * self.intermediate_size);
        defer allocator.free(ffn);

        for (self.layers) |layer| {
            layerNormPatches(norm, current, tokens, hidden, layer.ln1_weight, layer.ln1_bias, self.epsilon);
            try linearBatch(qkv, norm, tokens, layer.qkv_weight, layer.qkv_bias, options);
            applyVisionMRopeToQkv(qkv, patches.grid_width, self.head_count, head_dim);
            try visionAttentionInto(attention, qkv, tokens, hidden, self.head_count, options, allocator);
            try linearBatch(projected, attention, tokens, layer.attn_out_weight, layer.attn_out_bias, options);
            addInto(current, current, projected);

            layerNormPatches(norm, current, tokens, hidden, layer.ln2_weight, layer.ln2_bias, self.epsilon);
            try linearGeluBatch(ffn, norm, tokens, layer.ffn_up_weight, layer.ffn_up_bias, options);
            try linearBatch(projected, ffn, tokens, layer.ffn_down_weight, layer.ffn_down_bias, options);
            addInto(current, current, projected);
        }

        return .{
            .allocator = allocator,
            .tokens = tokens,
            .dimensions = hidden,
            .grid_width = patches.grid_width,
            .grid_height = patches.grid_height,
            .data = current,
        };
    }
};

pub const PatchProjector = struct {
    allocator: Allocator,
    norm_weight: ?tensor.Tensor,
    norm_bias: ?tensor.Tensor,
    fc1_weight: tensor.Tensor,
    fc1_bias: tensor.Tensor,
    fc2_weight: tensor.Tensor,
    fc2_bias: tensor.Tensor,
    spatial_merge_size: usize,
    epsilon: f32,

    pub fn deinit(self: *PatchProjector) void {
        if (self.norm_weight) |*norm_weight| norm_weight.deinit(self.allocator);
        if (self.norm_bias) |*norm_bias| norm_bias.deinit(self.allocator);
        self.fc1_weight.deinit(self.allocator);
        self.fc1_bias.deinit(self.allocator);
        self.fc2_weight.deinit(self.allocator);
        self.fc2_bias.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn forward(
        self: *const PatchProjector,
        allocator: Allocator,
        encoded: *const ImageEmbeddings,
        options: ProjectorOptions,
    ) !ImageEmbeddings {
        const merge = self.spatial_merge_size;
        if (merge == 0 or encoded.grid_width % merge != 0 or encoded.grid_height % merge != 0) {
            return Error.InvalidProjectorTensor;
        }

        const merge_area = merge * merge;
        const merged_tokens = encoded.tokens / merge_area;
        const merged_dim = encoded.dimensions * merge_area;
        const hidden = matrixRows(self.fc1_weight);
        const projected_dim = matrixRows(self.fc2_weight);
        if (matrixCols(self.fc1_weight) != merged_dim) return Error.InvalidProjectorTensor;
        if (self.fc1_bias.len() != hidden) return Error.InvalidProjectorTensor;
        if (matrixCols(self.fc2_weight) != hidden) return Error.InvalidProjectorTensor;
        if (self.fc2_bias.len() != projected_dim) return Error.InvalidProjectorTensor;
        if (self.norm_weight) |norm_weight| {
            if (norm_weight.len() != encoded.dimensions) return Error.InvalidProjectorTensor;
        }
        if (self.norm_bias) |norm_bias| {
            if (norm_bias.len() != encoded.dimensions) return Error.InvalidProjectorTensor;
        }

        const normalized = try allocator.alignedAlloc(f32, tensor.data_alignment, encoded.data.len);
        defer allocator.free(normalized);
        layerNormPatches(normalized, encoded.data, encoded.tokens, encoded.dimensions, self.norm_weight, self.norm_bias, self.epsilon);

        const merged = try allocator.alignedAlloc(f32, tensor.data_alignment, merged_tokens * merged_dim);
        defer allocator.free(merged);
        mergePatchGrid(merged, normalized, encoded.grid_width, encoded.grid_height, encoded.dimensions, merge);

        const hidden_values = try allocator.alignedAlloc(f32, tensor.data_alignment, merged_tokens * hidden);
        defer allocator.free(hidden_values);
        try linearGeluBatch(hidden_values, merged, merged_tokens, self.fc1_weight, self.fc1_bias, options);

        const out = try allocator.alignedAlloc(f32, tensor.data_alignment, merged_tokens * projected_dim);
        errdefer allocator.free(out);
        try linearBatch(out, hidden_values, merged_tokens, self.fc2_weight, self.fc2_bias, options);

        return .{
            .allocator = allocator,
            .tokens = merged_tokens,
            .dimensions = projected_dim,
            .grid_width = encoded.grid_width / merge,
            .grid_height = encoded.grid_height / merge,
            .data = out,
        };
    }
};

pub const MultimodalPrompt = struct {
    allocator: Allocator,
    rendered: []u8,
    tokens: []u32,
    image_token_id: u32,
    image_start: usize,
    image_token_count: usize,

    pub fn deinit(self: *MultimodalPrompt) void {
        self.allocator.free(self.rendered);
        self.allocator.free(self.tokens);
        self.* = undefined;
    }
};

const PatchWeightLayout = enum {
    ggml_hwio,
    ggml_oihw,
};

pub fn decodeResizeForPlan(allocator: Allocator, plan: NativePlan) !vision.RgbImage {
    const size = planImageSize(plan);
    return vision.decodeResizeRgbF32(allocator, plan.image, size, size, plan.image_mean, plan.image_std);
}

pub fn loadPatchEmbeddingFromFile(
    allocator: Allocator,
    io: std.Io,
    projector_path: []const u8,
    plan: NativePlan,
) !PatchEmbedding {
    var file = try std.Io.Dir.cwd().openFile(io, projector_path, .{ .mode = .read_only });
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

    var weight = try loadFirstTensorF32(allocator, parsed, map.memory, &.{ "v.patch_embd.weight", "v.patch_embd" });
    errdefer weight.deinit(allocator);

    var second_weight: ?tensor.Tensor = loadFirstTensorF32(allocator, parsed, map.memory, &.{"v.patch_embd.weight.1"}) catch |err| switch (err) {
        error.MissingPatchEmbedding => null,
        else => return err,
    };
    errdefer if (second_weight) |*owned_second_weight| owned_second_weight.deinit(allocator);

    var bias: ?tensor.Tensor = loadFirstTensorF32(allocator, parsed, map.memory, &.{ "v.patch_embd.bias", "v.patch_embd.bias.0" }) catch |err| switch (err) {
        error.MissingPatchEmbedding => null,
        else => return err,
    };
    errdefer if (bias) |*owned_bias| owned_bias.deinit(allocator);

    const dims = weight.dims();
    if (dims.len != 4) return Error.InvalidProjectorTensor;
    const patch = plan.patch_size orelse detectPatchSize(dims) orelse return Error.InvalidProjectorTensor;

    var layout: PatchWeightLayout = undefined;
    const in_channels: usize = 3;
    var out_channels: usize = undefined;
    if (dims[0] == patch and dims[1] == patch and dims[2] == 3) {
        layout = .ggml_hwio;
        out_channels = dims[3];
    } else if (dims[1] == 3 and dims[2] == patch and dims[3] == patch) {
        layout = .ggml_oihw;
        out_channels = dims[0];
    } else {
        return Error.InvalidProjectorTensor;
    }

    if (second_weight) |second_weight_tensor| {
        if (!std.mem.eql(usize, dims, second_weight_tensor.dims())) return Error.InvalidProjectorTensor;
    }
    if (bias) |bias_tensor| {
        if (bias_tensor.len() != out_channels) return Error.InvalidProjectorTensor;
    }

    return .{
        .allocator = allocator,
        .weight = weight,
        .second_weight = second_weight,
        .bias = bias,
        .layout = layout,
        .patch_size = patch,
        .in_channels = in_channels,
        .out_channels = out_channels,
    };
}

pub fn loadProjectorFromFile(
    allocator: Allocator,
    io: std.Io,
    projector_path: []const u8,
    plan: NativePlan,
) !PatchProjector {
    if (!std.mem.eql(u8, plan.projector_type, "qwen3vl_merger") and
        !std.mem.eql(u8, plan.projector_type, "qwen2vl_merger") and
        !std.mem.eql(u8, plan.projector_type, "qwen2.5vl_merger"))
    {
        return Error.UnsupportedProjectorArchitecture;
    }

    var file = try std.Io.Dir.cwd().openFile(io, projector_path, .{ .mode = .read_only });
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

    var norm_weight: ?tensor.Tensor = loadFirstTensorF32(allocator, parsed, map.memory, &.{ "v.post_ln.weight", "v.norm.weight" }) catch |err| switch (err) {
        error.MissingPatchEmbedding => null,
        else => return err,
    };
    errdefer if (norm_weight) |*owned_norm_weight| owned_norm_weight.deinit(allocator);

    var norm_bias: ?tensor.Tensor = loadFirstTensorF32(allocator, parsed, map.memory, &.{ "v.post_ln.bias", "v.norm.bias" }) catch |err| switch (err) {
        error.MissingPatchEmbedding => null,
        else => return err,
    };
    errdefer if (norm_bias) |*owned_norm_bias| owned_norm_bias.deinit(allocator);

    var fc1_weight = try loadNamedProjectorTensorF32(allocator, parsed, map.memory, "mm.0.weight");
    errdefer fc1_weight.deinit(allocator);
    var fc1_bias = try loadNamedProjectorTensorF32(allocator, parsed, map.memory, "mm.0.bias");
    errdefer fc1_bias.deinit(allocator);
    var fc2_weight = try loadNamedProjectorTensorF32(allocator, parsed, map.memory, "mm.2.weight");
    errdefer fc2_weight.deinit(allocator);
    var fc2_bias = try loadNamedProjectorTensorF32(allocator, parsed, map.memory, "mm.2.bias");
    errdefer fc2_bias.deinit(allocator);

    if (fc1_weight.rank() != 2 or fc2_weight.rank() != 2 or fc1_bias.rank() != 1 or fc2_bias.rank() != 1) {
        return Error.InvalidProjectorTensor;
    }
    if (fc1_bias.len() != matrixRows(fc1_weight) or fc2_bias.len() != matrixRows(fc2_weight)) {
        return Error.InvalidProjectorTensor;
    }

    return .{
        .allocator = allocator,
        .norm_weight = norm_weight,
        .norm_bias = norm_bias,
        .fc1_weight = fc1_weight,
        .fc1_bias = fc1_bias,
        .fc2_weight = fc2_weight,
        .fc2_bias = fc2_bias,
        .spatial_merge_size = plan.spatial_merge_size orelse 2,
        .epsilon = plan.layer_norm_epsilon orelse 0.000001,
    };
}

pub fn loadQwen3VisionTransformerFromFile(
    allocator: Allocator,
    io: std.Io,
    projector_path: []const u8,
    plan: NativePlan,
) !Qwen3VisionTransformer {
    if (!std.mem.eql(u8, plan.projector_type, "qwen3vl_merger")) {
        return Error.UnsupportedProjectorArchitecture;
    }

    var file = try std.Io.Dir.cwd().openFile(io, projector_path, .{ .mode = .read_only });
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

    var position_embeddings = try loadNamedVisionTensorF32(allocator, parsed, map.memory, "v.position_embd.weight");
    errdefer position_embeddings.deinit(allocator);

    const hidden = plan.embedding_length orelse matrixCols(position_embeddings);
    const intermediate = parsed.usizeValue("clip.vision.feed_forward_length") orelse return Error.InvalidProjectorTensor;
    const heads = parsed.usizeValue("clip.vision.attention.head_count") orelse return Error.InvalidProjectorTensor;
    const layer_count = plan.block_count orelse return Error.InvalidProjectorTensor;

    if (matrixCols(position_embeddings) != hidden) return Error.InvalidProjectorTensor;

    const layers = try allocator.alloc(VisionLayer, layer_count);
    errdefer allocator.free(layers);
    var loaded: usize = 0;
    errdefer {
        for (layers[0..loaded]) |*layer| layer.deinit(allocator);
    }

    var name_buf: [96]u8 = undefined;
    for (layers, 0..) |*layer, i| {
        layer.* = try loadVisionLayer(allocator, parsed, map.memory, &name_buf, i);
        loaded += 1;

        if (layer.ln1_weight.len() != hidden or layer.ln1_bias.len() != hidden) return Error.InvalidProjectorTensor;
        if (matrixRows(layer.qkv_weight) != hidden * 3 or matrixCols(layer.qkv_weight) != hidden) return Error.InvalidProjectorTensor;
        if (layer.qkv_bias.len() != hidden * 3) return Error.InvalidProjectorTensor;
        if (matrixRows(layer.attn_out_weight) != hidden or matrixCols(layer.attn_out_weight) != hidden) return Error.InvalidProjectorTensor;
        if (layer.attn_out_bias.len() != hidden) return Error.InvalidProjectorTensor;
        if (layer.ln2_weight.len() != hidden or layer.ln2_bias.len() != hidden) return Error.InvalidProjectorTensor;
        if (matrixRows(layer.ffn_up_weight) != intermediate or matrixCols(layer.ffn_up_weight) != hidden) return Error.InvalidProjectorTensor;
        if (layer.ffn_up_bias.len() != intermediate) return Error.InvalidProjectorTensor;
        if (matrixRows(layer.ffn_down_weight) != hidden or matrixCols(layer.ffn_down_weight) != intermediate) return Error.InvalidProjectorTensor;
        if (layer.ffn_down_bias.len() != hidden) return Error.InvalidProjectorTensor;
    }

    return .{
        .allocator = allocator,
        .position_embeddings = position_embeddings,
        .layers = layers,
        .hidden_size = hidden,
        .intermediate_size = intermediate,
        .head_count = heads,
        .epsilon = plan.layer_norm_epsilon orelse 0.000001,
    };
}

pub fn buildQwenImagePrompt(
    allocator: Allocator,
    tokenizer: *const tokenizer_mod.Tokenizer,
    prompt: []const u8,
    image_token_count: usize,
) !MultimodalPrompt {
    const image_token_id = tokenizer.tokenId("<|image_pad|>") orelse return Error.MissingImagePadToken;
    const prefix = "<|im_start|>user\n<|vision_start|>";
    const marker = "<|image_pad|>";
    const suffix_prefix = "<|vision_end|>\n";
    const suffix_postfix = "<|im_end|>\n<|im_start|>assistant\n";

    var rendered_list = std.array_list.Managed(u8).init(allocator);
    errdefer rendered_list.deinit();
    try rendered_list.appendSlice(prefix);
    for (0..image_token_count) |_| try rendered_list.appendSlice(marker);
    try rendered_list.appendSlice(suffix_prefix);
    try rendered_list.appendSlice(prompt);
    try rendered_list.appendSlice(suffix_postfix);
    const rendered = try rendered_list.toOwnedSlice();
    errdefer allocator.free(rendered);

    const tokens = try tokenizer.encode(allocator, rendered);
    errdefer allocator.free(tokens);

    var image_start: ?usize = null;
    var seen: usize = 0;
    for (tokens, 0..) |token, i| {
        if (token == image_token_id) {
            if (image_start == null) image_start = i;
            seen += 1;
        }
    }
    if (seen != image_token_count) return Error.MissingImagePadToken;

    return .{
        .allocator = allocator,
        .rendered = rendered,
        .tokens = tokens,
        .image_token_id = image_token_id,
        .image_start = image_start orelse return Error.MissingImagePadToken,
        .image_token_count = image_token_count,
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
    for (file.tensors) |info| {
        if (info.tensor_type.isNativeFloat()) count += 1;
    }
    return count;
}

fn imageArray3OrDefault(allocator: Allocator, file: gguf.File, key: []const u8, default: [3]f32) ![3]f32 {
    const values = file.f32ArrayAlloc(allocator, key) catch return default;
    defer allocator.free(values);
    if (values.len < 3) return default;
    return .{ values[0], values[1], values[2] };
}

fn planImageSize(plan: NativePlan) usize {
    const base = plan.image_size orelse 448;
    const patch = plan.patch_size orelse 16;
    const requested = if (base == 0) 448 else base;
    if (patch == 0) return requested;
    const remainder = requested % patch;
    return if (remainder == 0) requested else requested + patch - remainder;
}

fn detectPatchSize(dims: []const usize) ?usize {
    if (dims.len != 4) return null;
    if (dims[0] == dims[1] and dims[2] == 3) return dims[0];
    if (dims[2] == dims[3] and dims[1] == 3) return dims[2];
    return null;
}

fn loadFirstTensorF32(allocator: Allocator, file: gguf.File, bytes: []const u8, names: []const []const u8) !tensor.Tensor {
    for (names) |name| {
        if (file.tensor(name)) |info| return loadTensorF32(allocator, file, bytes, info);
    }
    return Error.MissingPatchEmbedding;
}

fn loadNamedProjectorTensorF32(allocator: Allocator, file: gguf.File, bytes: []const u8, name: []const u8) !tensor.Tensor {
    const info = file.tensor(name) orelse return Error.MissingProjectorTensor;
    return loadTensorF32(allocator, file, bytes, info);
}

fn loadNamedVisionTensorF32(allocator: Allocator, file: gguf.File, bytes: []const u8, name: []const u8) !tensor.Tensor {
    const info = file.tensor(name) orelse return Error.MissingVisionTensor;
    return loadTensorF32(allocator, file, bytes, info);
}

fn loadFormattedVisionTensorF32(
    allocator: Allocator,
    file: gguf.File,
    bytes: []const u8,
    name_buf: []u8,
    comptime format: []const u8,
    args: anytype,
) !tensor.Tensor {
    const name = try std.fmt.bufPrint(name_buf, format, args);
    return loadNamedVisionTensorF32(allocator, file, bytes, name);
}

fn loadVisionLayer(
    allocator: Allocator,
    file: gguf.File,
    bytes: []const u8,
    name_buf: []u8,
    index: usize,
) !VisionLayer {
    var layer = VisionLayer{
        .ln1_weight = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ln1.weight", .{index}),
        .ln1_bias = undefined,
        .qkv_weight = undefined,
        .qkv_bias = undefined,
        .attn_out_weight = undefined,
        .attn_out_bias = undefined,
        .ln2_weight = undefined,
        .ln2_bias = undefined,
        .ffn_up_weight = undefined,
        .ffn_up_bias = undefined,
        .ffn_down_weight = undefined,
        .ffn_down_bias = undefined,
    };
    errdefer layer.ln1_weight.deinit(allocator);

    layer.ln1_bias = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ln1.bias", .{index});
    errdefer layer.ln1_bias.deinit(allocator);
    layer.qkv_weight = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.attn_qkv.weight", .{index});
    errdefer layer.qkv_weight.deinit(allocator);
    layer.qkv_bias = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.attn_qkv.bias", .{index});
    errdefer layer.qkv_bias.deinit(allocator);
    layer.attn_out_weight = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.attn_out.weight", .{index});
    errdefer layer.attn_out_weight.deinit(allocator);
    layer.attn_out_bias = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.attn_out.bias", .{index});
    errdefer layer.attn_out_bias.deinit(allocator);
    layer.ln2_weight = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ln2.weight", .{index});
    errdefer layer.ln2_weight.deinit(allocator);
    layer.ln2_bias = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ln2.bias", .{index});
    errdefer layer.ln2_bias.deinit(allocator);
    layer.ffn_up_weight = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ffn_up.weight", .{index});
    errdefer layer.ffn_up_weight.deinit(allocator);
    layer.ffn_up_bias = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ffn_up.bias", .{index});
    errdefer layer.ffn_up_bias.deinit(allocator);
    layer.ffn_down_weight = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ffn_down.weight", .{index});
    errdefer layer.ffn_down_weight.deinit(allocator);
    layer.ffn_down_bias = try loadFormattedVisionTensorF32(allocator, file, bytes, name_buf, "v.blk.{d}.ffn_down.bias", .{index});

    return layer;
}

fn loadTensorF32(allocator: Allocator, file: gguf.File, bytes: []const u8, info: gguf.TensorInfo) !tensor.Tensor {
    if (!info.tensor_type.isNativeFloat()) return Error.UnsupportedProjectorDType;

    var dims_buf: [8]usize = undefined;
    for (info.shape(), 0..) |dim, i| dims_buf[i] = @intCast(dim);
    const dim_count: usize = @intCast(info.dim_count);
    var out = try tensor.Tensor.initUndefined(allocator, dims_buf[0..dim_count]);
    errdefer out.deinit(allocator);

    const raw = try tensorBytes(file, bytes, info);
    switch (info.tensor_type) {
        .f32 => {
            if (raw.len != out.data.len * @sizeOf(f32)) return Error.InvalidProjectorTensor;
            for (out.data, 0..) |*dst, i| {
                const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
                dst.* = @bitCast(bits);
            }
        },
        .f16 => {
            if (raw.len != out.data.len * @sizeOf(u16)) return Error.InvalidProjectorTensor;
            for (out.data, 0..) |*dst, i| {
                const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                dst.* = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .bf16 => {
            if (raw.len != out.data.len * @sizeOf(u16)) return Error.InvalidProjectorTensor;
            for (out.data, 0..) |*dst, i| {
                const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                dst.* = bf16ToF32(bits);
            }
        },
        else => return Error.UnsupportedProjectorDType,
    }
    return out;
}

fn tensorBytes(file: gguf.File, bytes: []const u8, info: gguf.TensorInfo) ![]const u8 {
    const elem_size: usize = switch (info.tensor_type) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        else => return Error.UnsupportedProjectorDType,
    };
    const byte_len = info.elementCount() * elem_size;
    const start = file.data_start + @as(usize, @intCast(info.offset));
    const end = start + byte_len;
    if (end > bytes.len or end < start) return Error.InvalidProjectorTensor;
    return bytes[start..end];
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn emptyAlignedF32() []align(tensor.data_alignment_bytes) f32 {
    return @as([*]align(tensor.data_alignment_bytes) f32, @ptrFromInt(tensor.data_alignment_bytes))[0..0];
}

fn matrixRows(weight: tensor.Tensor) usize {
    const dims = weight.dims();
    std.debug.assert(dims.len == 2);
    return dims[1];
}

fn matrixCols(weight: tensor.Tensor) usize {
    const dims = weight.dims();
    std.debug.assert(dims.len == 2);
    return dims[0];
}

fn layerNormPatches(
    out: []f32,
    input: []const f32,
    tokens: usize,
    dimensions: usize,
    weight: ?tensor.Tensor,
    bias: ?tensor.Tensor,
    epsilon: f32,
) void {
    std.debug.assert(out.len == input.len);
    for (0..tokens) |token_index| {
        const start = token_index * dimensions;
        const src = input[start..][0..dimensions];
        const dst = out[start..][0..dimensions];

        var sum: f32 = 0.0;
        for (src) |value| sum += value;
        const mean = sum / @as(f32, @floatFromInt(dimensions));

        var variance_sum: f32 = 0.0;
        for (src) |value| {
            const centered = value - mean;
            variance_sum = @mulAdd(f32, centered, centered, variance_sum);
        }
        const scale = 1.0 / @sqrt(variance_sum / @as(f32, @floatFromInt(dimensions)) + epsilon);

        for (dst, 0..) |*item, i| {
            var value = (src[i] - mean) * scale;
            if (weight) |norm_weight| value *= norm_weight.data[i];
            if (bias) |norm_bias| value += norm_bias.data[i];
            item.* = value;
        }
    }
}

fn mergePatchGrid(
    out: []f32,
    input: []const f32,
    grid_width: usize,
    grid_height: usize,
    dimensions: usize,
    merge: usize,
) void {
    const out_width = grid_width / merge;
    const out_height = grid_height / merge;
    const merged_dim = dimensions * merge * merge;
    std.debug.assert(out.len == out_width * out_height * merged_dim);

    var dst_token: usize = 0;
    for (0..out_height) |oy| {
        for (0..out_width) |ox| {
            var dst_offset = dst_token * merged_dim;
            for (0..merge) |dy| {
                for (0..merge) |dx| {
                    const src_token = (oy * merge + dy) * grid_width + (ox * merge + dx);
                    const src = input[src_token * dimensions ..][0..dimensions];
                    @memcpy(out[dst_offset..][0..dimensions], src);
                    dst_offset += dimensions;
                }
            }
            dst_token += 1;
        }
    }
}

const LinearBatchJob = struct {
    out: []f32,
    input: []const f32,
    token_count: usize,
    weight: tensor.Tensor,
    bias: tensor.Tensor,
    start_token: usize,
    end_token: usize,
    gelu: bool,
};

const PatchEmbeddingJob = struct {
    patch: *const PatchEmbedding,
    image: *const vision.RgbImage,
    data: []f32,
    out_w: usize,
    start_token: usize,
    end_token: usize,
};

fn patchEmbeddingWorker(job: *const PatchEmbeddingJob) void {
    job.patch.forwardTokenRange(job.image, job.data, job.out_w, job.start_token, job.end_token);
}

fn linearBatch(
    out: []f32,
    input: []const f32,
    token_count: usize,
    weight: tensor.Tensor,
    bias: tensor.Tensor,
    options: ProjectorOptions,
) !void {
    return linearBatchImpl(out, input, token_count, weight, bias, options, false);
}

fn linearGeluBatch(
    out: []f32,
    input: []const f32,
    token_count: usize,
    weight: tensor.Tensor,
    bias: tensor.Tensor,
    options: ProjectorOptions,
) !void {
    return linearBatchImpl(out, input, token_count, weight, bias, options, true);
}

fn linearBatchImpl(
    out: []f32,
    input: []const f32,
    token_count: usize,
    weight: tensor.Tensor,
    bias: tensor.Tensor,
    options: ProjectorOptions,
    gelu: bool,
) !void {
    const rows = matrixRows(weight);
    const cols = matrixCols(weight);
    std.debug.assert(input.len == token_count * cols);
    std.debug.assert(out.len == token_count * rows);
    std.debug.assert(bias.len() == rows);

    const workers = resolveProjectorWorkers(token_count, options);
    if (workers == 1) {
        linearBatchRange(out, input, weight, bias, 0, token_count, gelu);
        return;
    }

    var threads: [32]std.Thread = undefined;
    var jobs: [32]LinearBatchJob = undefined;
    const tokens_per_worker = (token_count + workers - 1) / workers;

    var spawned: usize = 0;
    while (spawned + 1 < workers) : (spawned += 1) {
        const start = spawned * tokens_per_worker;
        const end = @min(start + tokens_per_worker, token_count);
        jobs[spawned] = .{
            .out = out,
            .input = input,
            .token_count = token_count,
            .weight = weight,
            .bias = bias,
            .start_token = start,
            .end_token = end,
            .gelu = gelu,
        };
        threads[spawned] = try std.Thread.spawn(.{}, linearBatchWorker, .{&jobs[spawned]});
    }

    const start = spawned * tokens_per_worker;
    linearBatchRange(out, input, weight, bias, start, token_count, gelu);
    for (threads[0..spawned]) |thread| thread.join();
}

fn linearBatchWorker(job: *const LinearBatchJob) void {
    linearBatchRange(job.out, job.input, job.weight, job.bias, job.start_token, job.end_token, job.gelu);
}

fn linearBatchRange(
    out: []f32,
    input: []const f32,
    weight: tensor.Tensor,
    bias: tensor.Tensor,
    start_token: usize,
    end_token: usize,
    gelu: bool,
) void {
    const rows = matrixRows(weight);
    const cols = matrixCols(weight);
    for (start_token..end_token) |token_index| {
        const src = input[token_index * cols ..][0..cols];
        const dst = out[token_index * rows ..][0..rows];
        for (0..rows) |row| {
            const weights = weight.data[row * cols ..][0..cols];
            var value = dotF32(weights, src) + bias.data[row];
            if (gelu) value = geluApprox(value);
            dst[row] = value;
        }
    }
}

fn resolveProjectorWorkers(token_count: usize, options: ProjectorOptions) usize {
    if (token_count <= 1) return 1;
    const requested = if (options.thread_count == 0) std.Thread.getCpuCount() catch 1 else options.thread_count;
    return @max(@as(usize, 1), @min(@min(requested, token_count), 32));
}

fn dotF32(a: []const f32, b: []const f32) f32 {
    return tensor.dotAssumeEqual(a, b);
}

fn geluApprox(x: f32) f32 {
    const sqrt_two_over_pi: f32 = 0.7978845608028654;
    const cubic = x * x * x;
    return 0.5 * x * (1.0 + std.math.tanh(sqrt_two_over_pi * (x + 0.044715 * cubic)));
}

fn addPositionEmbeddings(values: []f32, position_embeddings: tensor.Tensor) void {
    const hidden = matrixCols(position_embeddings);
    const tokens = matrixRows(position_embeddings);
    std.debug.assert(values.len == tokens * hidden);
    addInto(values, values, position_embeddings.data);
}

fn addInto(out: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(out.len == a.len and out.len == b.len);
    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        const av: Vec = a[i..][0..simd_lanes].*;
        const bv: Vec = b[i..][0..simd_lanes].*;
        out[i..][0..simd_lanes].* = av + bv;
    }
    while (i < out.len) : (i += 1) out[i] = a[i] + b[i];
}

fn applyVisionMRopeToQkv(qkv: []f32, grid_width: usize, heads: usize, head_dim: usize) void {
    const hidden = heads * head_dim;
    const qkv_dim = hidden * 3;
    const tokens = qkv.len / qkv_dim;
    for (0..tokens) |token_index| {
        const y: f32 = @floatFromInt(token_index / grid_width);
        const x: f32 = @floatFromInt(token_index % grid_width);
        const positions = [4]f32{ y, x, y, x };
        const base = token_index * qkv_dim;
        for (0..heads) |head| {
            const offset = head * head_dim;
            applyVisionMRopeHead(qkv[base + offset ..][0..head_dim], positions);
            applyVisionMRopeHead(qkv[base + hidden + offset ..][0..head_dim], positions);
        }
    }
}

fn applyVisionMRopeHead(values: []f32, positions: [4]f32) void {
    const section_dim = values.len / 4;
    if (section_dim < 2) return;
    const theta: f32 = 10_000.0;
    for (0..4) |section| {
        const start = section * section_dim;
        const end = if (section == 3) values.len else start + section_dim;
        const dim_f: f32 = @floatFromInt(end - start);
        var i = start;
        while (i + 1 < end) : (i += 2) {
            const local: f32 = @floatFromInt(i - start);
            const inv_freq = 1.0 / std.math.pow(f32, theta, local / dim_f);
            const angle = positions[section] * inv_freq;
            const cos_v = std.math.cos(angle);
            const sin_v = std.math.sin(angle);
            const x0 = values[i];
            const x1 = values[i + 1];
            values[i] = x0 * cos_v - x1 * sin_v;
            values[i + 1] = x0 * sin_v + x1 * cos_v;
        }
    }
}

const VisionAttentionJob = struct {
    out: []f32,
    qkv: []const f32,
    tokens: usize,
    hidden: usize,
    head_dim: usize,
    head_start: usize,
    head_end: usize,
    scores: []f32,
};

fn visionAttentionInto(
    out: []f32,
    qkv: []const f32,
    tokens: usize,
    hidden: usize,
    heads: usize,
    options: ProjectorOptions,
    allocator: Allocator,
) !void {
    std.debug.assert(out.len == tokens * hidden);
    std.debug.assert(qkv.len == tokens * hidden * 3);
    @memset(out, 0.0);

    const workers = resolveAttentionWorkers(heads, options);
    if (workers == 1) {
        const scores = try allocator.alloc(f32, tokens);
        defer allocator.free(scores);
        visionAttentionHeadRange(out, qkv, tokens, hidden, hidden / heads, 0, heads, scores);
        return;
    }

    const scores = try allocator.alloc(f32, workers * tokens);
    defer allocator.free(scores);

    var threads: [32]std.Thread = undefined;
    var jobs: [32]VisionAttentionJob = undefined;
    const heads_per_worker = (heads + workers - 1) / workers;

    var spawned: usize = 0;
    while (spawned + 1 < workers) : (spawned += 1) {
        const start = spawned * heads_per_worker;
        const end = @min(start + heads_per_worker, heads);
        jobs[spawned] = .{
            .out = out,
            .qkv = qkv,
            .tokens = tokens,
            .hidden = hidden,
            .head_dim = hidden / heads,
            .head_start = start,
            .head_end = end,
            .scores = scores[spawned * tokens ..][0..tokens],
        };
        threads[spawned] = try std.Thread.spawn(.{}, visionAttentionWorker, .{&jobs[spawned]});
    }

    const start = spawned * heads_per_worker;
    visionAttentionHeadRange(out, qkv, tokens, hidden, hidden / heads, start, heads, scores[spawned * tokens ..][0..tokens]);
    for (threads[0..spawned]) |thread| thread.join();
}

fn resolveAttentionWorkers(heads: usize, options: ProjectorOptions) usize {
    if (heads <= 1) return 1;
    const requested = if (options.thread_count == 0) std.Thread.getCpuCount() catch 1 else options.thread_count;
    return @max(@as(usize, 1), @min(@min(requested, heads), 32));
}

fn visionAttentionWorker(job: *const VisionAttentionJob) void {
    visionAttentionHeadRange(job.out, job.qkv, job.tokens, job.hidden, job.head_dim, job.head_start, job.head_end, job.scores);
}

fn visionAttentionHeadRange(
    out: []f32,
    qkv: []const f32,
    tokens: usize,
    hidden: usize,
    head_dim: usize,
    head_start: usize,
    head_end: usize,
    scores: []f32,
) void {
    const qkv_dim = hidden * 3;
    const inv_sqrt_head = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (head_start..head_end) |head| {
        const head_offset = head * head_dim;
        for (0..tokens) |query_token| {
            const query = qkv[query_token * qkv_dim + head_offset ..][0..head_dim];
            for (scores[0..tokens], 0..) |*score, key_token| {
                const key = qkv[key_token * qkv_dim + hidden + head_offset ..][0..head_dim];
                score.* = dotF32(query, key) * inv_sqrt_head;
            }
            softmaxInPlace(scores[0..tokens]);

            const dst = out[query_token * hidden + head_offset ..][0..head_dim];
            @memset(dst, 0.0);
            for (scores[0..tokens], 0..) |score, value_token| {
                const value = qkv[value_token * qkv_dim + 2 * hidden + head_offset ..][0..head_dim];
                addScaled(dst, value, score);
            }
        }
    }
}

fn addScaled(out: []f32, values: []const f32, scale: f32) void {
    std.debug.assert(out.len == values.len);
    const scale_v: Vec = @splat(scale);
    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        const ov: Vec = out[i..][0..simd_lanes].*;
        const vv: Vec = values[i..][0..simd_lanes].*;
        out[i..][0..simd_lanes].* = ov + vv * scale_v;
    }
    while (i < out.len) : (i += 1) out[i] = @mulAdd(f32, values[i], scale, out[i]);
}

fn softmaxInPlace(values: []f32) void {
    var max_value = values[0];
    for (values[1..]) |value| max_value = @max(max_value, value);

    var sum: f32 = 0.0;
    for (values) |*value| {
        value.* = std.math.exp(value.* - max_value);
        sum += value.*;
    }
    const inv_sum = 1.0 / sum;
    for (values) |*value| value.* *= inv_sum;
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
        .spatial_merge_size = 2,
        .layer_norm_epsilon = 0.000001,
        .projector_tensor_count = 2,
        .projector_native_tensors = 2,
        .projector_quantized_tensors = 0,
        .image_mean = .{ 0.0, 0.0, 0.0 },
        .image_std = .{ 1.0, 1.0, 1.0 },
    };
    plan.deinit();
}

test "patch embedding forward uses contiguous patch-major output" {
    const allocator = std.testing.allocator;
    var image = vision.RgbImage{
        .allocator = allocator,
        .width = 2,
        .height = 2,
        .data = try allocator.alignedAlloc(f32, .@"64", 12),
    };
    defer image.deinit();
    const pixels = [_]f32{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    @memcpy(image.data, pixels[0..]);

    var weight = try tensor.Tensor.fromSlice(allocator, &.{ 1, 1, 3, 2 }, &.{
        1,  2,  3,
        10, 20, 30,
    });
    errdefer weight.deinit(allocator);
    var patch = PatchEmbedding{
        .allocator = allocator,
        .weight = weight,
        .second_weight = null,
        .bias = null,
        .layout = .ggml_hwio,
        .patch_size = 1,
        .in_channels = 3,
        .out_channels = 2,
    };
    defer patch.deinit();

    var embeddings = try patch.forward(allocator, &image);
    defer embeddings.deinit();

    try std.testing.expectEqual(@as(usize, 4), embeddings.tokens);
    try std.testing.expectEqual(@as(usize, 2), embeddings.dimensions);
    try std.testing.expectEqual(@as(usize, 2), embeddings.grid_width);
    try std.testing.expectEqual(@as(usize, 2), embeddings.grid_height);
    try std.testing.expectEqualSlices(f32, &.{ 14, 140, 32, 320, 50, 500, 68, 680 }, embeddings.data);
}

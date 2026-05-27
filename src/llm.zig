// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const gguf = @import("gguf");
const tensor = @import("tensor");

pub const Allocator = std.mem.Allocator;
pub const data_alignment_bytes = tensor.data_alignment_bytes;
pub const data_alignment = tensor.data_alignment;
pub const simd_lanes = tensor.simd_lanes;
pub const max_worker_threads = 64;

const Vec = @Vector(simd_lanes, f32);
const U16Vec = @Vector(simd_lanes, u16);
const U32Vec = @Vector(simd_lanes, u32);

pub const Error = tensor.Error || error{
    ContextFull,
    InvalidConfig,
    InvalidToken,
    InvalidWeight,
    MissingJsonField,
    MissingSafetensorsTensor,
    UnsupportedDType,
    UnsupportedModelArchitecture,
    ThreadSpawnFailed,
};

pub const Architecture = enum {
    llama,
    mistral,
    olmoe,
    qwen2,

    pub fn ggufPrefix(self: Architecture) []const u8 {
        return switch (self) {
            .llama => "llama",
            .mistral => "mistral",
            .olmoe => "olmoe",
            .qwen2 => "qwen2",
        };
    }
};

pub const FeedForwardKind = enum {
    dense,
    moe,
};

pub const RopeLayout = enum {
    interleaved,
    split_half,
};

pub const WeightKind = enum {
    f32,
    bf16,
};

pub const RuntimeOptions = struct {
    /// 0 means use the host CPU count. Each call dynamically lowers this based
    /// on matrix row count so small projections stay single-threaded.
    thread_count: usize = 0,
    min_rows_per_thread: usize = 64,
    preload_layers_ahead: usize = 1,
};

pub const LoadProgress = struct {
    context: ?*anyopaque = null,
    on_step: ?*const fn (?*anyopaque, completed: usize, total: usize, name: []const u8) void = null,

    pub fn report(self: LoadProgress, completed: usize, total: usize, name: []const u8) void {
        if (self.on_step) |callback| callback(self.context, completed, total, name);
    }
};

pub const Config = struct {
    architecture: Architecture = .qwen2,
    feed_forward: FeedForwardKind = .dense,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_experts: usize = 0,
    num_experts_per_token: usize = 0,
    qk_norm: bool = false,
    rope_layout: RopeLayout = .split_half,
    max_position_embeddings: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    qkv_bias: bool = true,
    eos_token_id: u32,
    pad_token_id: u32,

    pub fn fromJson(allocator: Allocator, json_bytes: []const u8) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{ .parse_numbers = true });
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return Error.InvalidConfig;
        const object = root.object;

        const model_type = getString(object, "model_type") orelse return Error.MissingJsonField;
        const architecture: Architecture = if (std.mem.eql(u8, model_type, "llama"))
            .llama
        else if (std.mem.eql(u8, model_type, "mistral"))
            .mistral
        else if (std.mem.eql(u8, model_type, "olmoe"))
            .olmoe
        else if (std.mem.eql(u8, model_type, "qwen2"))
            .qwen2
        else
            return Error.UnsupportedModelArchitecture;

        const heads = try getRequiredUsize(object, "num_attention_heads");
        const num_experts = getOptionalUsize(object, "num_experts") orelse 0;
        const feed_forward: FeedForwardKind = if (num_experts == 0) .dense else .moe;
        const default_qkv_bias = architecture == .qwen2;
        var config = Config{
            .architecture = architecture,
            .feed_forward = feed_forward,
            .vocab_size = try getRequiredUsize(object, "vocab_size"),
            .hidden_size = try getRequiredUsize(object, "hidden_size"),
            .intermediate_size = try getRequiredUsize(object, "intermediate_size"),
            .num_hidden_layers = try getRequiredUsize(object, "num_hidden_layers"),
            .num_attention_heads = heads,
            .num_key_value_heads = getOptionalUsize(object, "num_key_value_heads") orelse heads,
            .num_experts = num_experts,
            .num_experts_per_token = getOptionalUsize(object, "num_experts_per_tok") orelse 0,
            .qk_norm = architecture == .olmoe,
            .rope_layout = .split_half,
            .max_position_embeddings = try getRequiredUsize(object, "max_position_embeddings"),
            .rope_theta = getOptionalF32(object, "rope_theta") orelse 10_000.0,
            .rms_norm_eps = getOptionalF32(object, "rms_norm_eps") orelse 0.000001,
            .tie_word_embeddings = getOptionalBool(object, "tie_word_embeddings") orelse false,
            .qkv_bias = getOptionalBool(object, "attention_bias") orelse default_qkv_bias,
            .eos_token_id = getOptionalTokenId(object, "eos_token_id") orelse 0,
            .pad_token_id = getOptionalTokenId(object, "pad_token_id") orelse 0,
        };
        try config.validate();
        return config;
    }

    pub fn loadFromFile(allocator: Allocator, io: std.Io, path: []const u8) !Config {
        const json_bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
        defer allocator.free(json_bytes);
        return fromJson(allocator, json_bytes);
    }

    pub fn fromGguf(file: gguf.File) !Config {
        const arch_text = file.string("general.architecture") orelse return Error.MissingJsonField;
        const architecture: Architecture = if (std.mem.eql(u8, arch_text, "llama"))
            .llama
        else if (std.mem.eql(u8, arch_text, "mistral"))
            .mistral
        else if (std.mem.eql(u8, arch_text, "qwen2"))
            .qwen2
        else if (std.mem.eql(u8, arch_text, "olmoe"))
            .olmoe
        else
            return Error.UnsupportedModelArchitecture;

        const prefix = architecture.ggufPrefix();
        const vocab = file.metadataValue("tokenizer.ggml.tokens") orelse return Error.MissingJsonField;
        const vocab_size = switch (vocab) {
            .array => |array| array.len,
            else => return Error.InvalidConfig,
        };
        const heads = try requiredGgufUsize(file, prefix, "attention.head_count");
        var config = Config{
            .architecture = architecture,
            .feed_forward = .dense,
            .vocab_size = vocab_size,
            .hidden_size = try requiredGgufUsize(file, prefix, "embedding_length"),
            .intermediate_size = try requiredGgufUsize(file, prefix, "feed_forward_length"),
            .num_hidden_layers = try requiredGgufUsize(file, prefix, "block_count"),
            .num_attention_heads = heads,
            .num_key_value_heads = optionalGgufUsize(file, prefix, "attention.head_count_kv") orelse heads,
            .rope_layout = .interleaved,
            .max_position_embeddings = try requiredGgufUsize(file, prefix, "context_length"),
            .rope_theta = optionalGgufF32(file, prefix, "rope.freq_base") orelse 10_000.0,
            .rms_norm_eps = optionalGgufF32(file, prefix, "attention.layer_norm_rms_epsilon") orelse 0.000001,
            .tie_word_embeddings = file.tensor("output.weight") == null,
            .qkv_bias = false,
            .eos_token_id = @intCast(file.usizeValue("tokenizer.ggml.eos_token_id") orelse 0),
            .pad_token_id = @intCast(file.usizeValue("tokenizer.ggml.padding_token_id") orelse 0),
        };
        try config.validate();
        return config;
    }

    pub fn validate(self: Config) Error!void {
        if (self.vocab_size == 0 or
            self.hidden_size == 0 or
            self.intermediate_size == 0 or
            self.num_hidden_layers == 0 or
            self.num_attention_heads == 0 or
            self.num_key_value_heads == 0 or
            self.max_position_embeddings == 0)
        {
            return Error.InvalidConfig;
        }
        if (self.hidden_size % self.num_attention_heads != 0) return Error.InvalidConfig;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return Error.InvalidConfig;
        if (self.headDim() % 2 != 0) return Error.InvalidConfig;
        switch (self.feed_forward) {
            .dense => if (self.num_experts != 0 or self.num_experts_per_token != 0) return Error.InvalidConfig,
            .moe => {
                if (self.num_experts == 0 or self.num_experts_per_token == 0) return Error.InvalidConfig;
                if (self.num_experts_per_token > self.num_experts) return Error.InvalidConfig;
            },
        }
    }

    pub fn supportsDenseRunner(self: Config) bool {
        return self.feed_forward == .dense and switch (self.architecture) {
            .llama, .mistral, .qwen2 => true,
            .olmoe => false,
        };
    }

    pub fn isMoe(self: Config) bool {
        return self.feed_forward == .moe;
    }

    pub fn headDim(self: Config) usize {
        return self.hidden_size / self.num_attention_heads;
    }

    pub fn kvDim(self: Config) usize {
        return self.num_key_value_heads * self.headDim();
    }

    pub fn kvGroupSize(self: Config) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }

    pub fn layerParameterCount(self: Config) usize {
        const hidden = self.hidden_size;
        const kv = self.kvDim();
        const inter = self.intermediate_size;
        const bias = if (self.qkv_bias) hidden + kv + kv else 0;

        var total = hidden + // input RMS norm
            hidden * hidden + // q_proj
            bias +
            kv * hidden + // k_proj
            kv * hidden + // v_proj
            hidden * hidden + // o_proj
            hidden; // post attention RMS norm
        if (self.qk_norm) total += self.headDim() * 2;
        switch (self.feed_forward) {
            .dense => {
                total += inter * hidden; // gate_proj
                total += inter * hidden; // up_proj
                total += hidden * inter; // down_proj
            },
            .moe => {
                total += self.num_experts * hidden; // router
                total += self.num_experts * (inter * hidden * 2 + hidden * inter);
            },
        }
        return total;
    }

    pub fn parameterCount(self: Config) usize {
        var total = self.vocab_size * self.hidden_size;
        total += self.num_hidden_layers * self.layerParameterCount();
        total += self.hidden_size;
        if (!self.tie_word_embeddings) total += self.vocab_size * self.hidden_size;
        return total;
    }
};

pub const WeightSlice = union(WeightKind) {
    f32: []const f32,
    bf16: []const u16,

    pub fn len(self: WeightSlice) usize {
        return switch (self) {
            .f32 => |values| values.len,
            .bf16 => |values| values.len,
        };
    }

    pub fn at(self: WeightSlice, index: usize) f32 {
        return switch (self) {
            .f32 => |values| values[index],
            .bf16 => |values| bf16ToF32(values[index]),
        };
    }

    pub fn prefetch(self: WeightSlice) void {
        switch (self) {
            .f32 => |values| if (values.len != 0) @prefetch(values.ptr, .{ .rw = .read, .locality = 3 }),
            .bf16 => |values| if (values.len != 0) @prefetch(values.ptr, .{ .rw = .read, .locality = 3 }),
        }
    }
};

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    values: WeightSlice,

    pub fn initF32(rows: usize, cols: usize, values: []const f32) Error!Matrix {
        if (rows == 0 or cols == 0 or rows * cols != values.len) return Error.InvalidWeight;
        return .{ .rows = rows, .cols = cols, .values = .{ .f32 = values } };
    }

    pub fn initBf16(rows: usize, cols: usize, values: []const u16) Error!Matrix {
        if (rows == 0 or cols == 0 or rows * cols != values.len) return Error.InvalidWeight;
        return .{ .rows = rows, .cols = cols, .values = .{ .bf16 = values } };
    }

    pub fn validate(self: Matrix) Error!void {
        if (self.rows == 0 or self.cols == 0 or self.rows * self.cols != self.values.len()) {
            return Error.InvalidWeight;
        }
    }

    pub fn dotRow(self: Matrix, row_index: usize, input: []const f32) f32 {
        std.debug.assert(row_index < self.rows);
        std.debug.assert(input.len == self.cols);

        const start = row_index * self.cols;
        return switch (self.values) {
            .f32 => |values| dotF32(values[start..][0..self.cols], input),
            .bf16 => |values| dotBf16(values[start..][0..self.cols], input),
        };
    }

    pub fn prefetch(self: Matrix) void {
        self.values.prefetch();
    }
};

pub const LayerWeights = struct {
    input_norm: WeightSlice,
    q_proj: Matrix,
    q_bias: ?WeightSlice = null,
    k_proj: Matrix,
    k_bias: ?WeightSlice = null,
    v_proj: Matrix,
    v_bias: ?WeightSlice = null,
    o_proj: Matrix,
    o_bias: ?WeightSlice = null,
    post_attention_norm: WeightSlice,
    gate_proj: Matrix,
    up_proj: Matrix,
    down_proj: Matrix,

    pub fn validate(self: LayerWeights, config: Config) Error!void {
        const hidden = config.hidden_size;
        const kv = config.kvDim();
        const inter = config.intermediate_size;

        try expectWeightLen(self.input_norm, hidden);
        try expectMatrix(self.q_proj, hidden, hidden);
        try expectMatrix(self.k_proj, kv, hidden);
        try expectMatrix(self.v_proj, kv, hidden);
        try expectMatrix(self.o_proj, hidden, hidden);
        try expectWeightLen(self.post_attention_norm, hidden);
        try expectMatrix(self.gate_proj, inter, hidden);
        try expectMatrix(self.up_proj, inter, hidden);
        try expectMatrix(self.down_proj, hidden, inter);
        if (config.qkv_bias) {
            if (self.q_bias) |bias| try expectWeightLen(bias, hidden) else return Error.InvalidWeight;
            if (self.k_bias) |bias| try expectWeightLen(bias, kv) else return Error.InvalidWeight;
            if (self.v_bias) |bias| try expectWeightLen(bias, kv) else return Error.InvalidWeight;
        } else {
            if (self.q_bias != null or self.k_bias != null or self.v_bias != null) return Error.InvalidWeight;
        }
        if (self.o_bias) |bias| try expectWeightLen(bias, hidden);
    }

    pub fn prefetch(self: LayerWeights) void {
        self.input_norm.prefetch();
        self.q_proj.prefetch();
        self.k_proj.prefetch();
        self.v_proj.prefetch();
        self.o_proj.prefetch();
        self.post_attention_norm.prefetch();
        self.gate_proj.prefetch();
        self.up_proj.prefetch();
        self.down_proj.prefetch();
    }
};

pub const ModelWeights = struct {
    config: Config,
    token_embedding: Matrix,
    layers: []const LayerWeights,
    final_norm: WeightSlice,
    lm_head: ?Matrix = null,

    pub fn validate(self: ModelWeights) Error!void {
        try self.config.validate();
        try expectMatrix(self.token_embedding, self.config.vocab_size, self.config.hidden_size);
        if (self.layers.len != self.config.num_hidden_layers) return Error.InvalidWeight;
        for (self.layers) |layer| try layer.validate(self.config);
        try expectWeightLen(self.final_norm, self.config.hidden_size);
        if (self.lm_head) |head| try expectMatrix(head, self.config.vocab_size, self.config.hidden_size);
        if (!self.config.tie_word_embeddings and self.lm_head == null) return Error.InvalidWeight;
    }

    pub fn logitsMatrix(self: ModelWeights) Matrix {
        return self.lm_head orelse self.token_embedding;
    }
};

pub const OwnedModel = struct {
    allocator: Allocator,
    config: Config,
    kind: WeightKind,
    storage_f32: []align(data_alignment_bytes) f32 = emptyAligned(f32),
    storage_bf16: []align(data_alignment_bytes) u16 = emptyAligned(u16),
    layers: []LayerWeights = &.{},
    weights: ModelWeights,

    pub fn init(allocator: Allocator, config: Config, kind: WeightKind) !OwnedModel {
        try config.validate();
        if (!config.supportsDenseRunner()) return Error.UnsupportedModelArchitecture;

        const storage_elements = storageElementCount(config, kind);
        var owned = OwnedModel{
            .allocator = allocator,
            .config = config,
            .kind = kind,
            .layers = try allocator.alloc(LayerWeights, config.num_hidden_layers),
            .weights = undefined,
        };
        errdefer allocator.free(owned.layers);

        switch (kind) {
            .f32 => owned.storage_f32 = try allocator.alignedAlloc(f32, data_alignment, storage_elements),
            .bf16 => owned.storage_bf16 = try allocator.alignedAlloc(u16, data_alignment, storage_elements),
        }
        errdefer owned.freeStorage();

        owned.assignSlices();
        return owned;
    }

    pub fn deinit(self: *OwnedModel) void {
        self.freeStorage();
        if (self.layers.len != 0) self.allocator.free(self.layers);
        self.* = .{
            .allocator = self.allocator,
            .config = self.config,
            .kind = self.kind,
            .weights = undefined,
        };
    }

    pub fn byteLen(self: *const OwnedModel) usize {
        return switch (self.kind) {
            .f32 => self.storage_f32.len * @sizeOf(f32),
            .bf16 => self.storage_bf16.len * @sizeOf(u16),
        };
    }

    pub fn fillDeterministicForTesting(self: *OwnedModel) void {
        switch (self.kind) {
            .f32 => for (self.storage_f32, 0..) |*value, i| {
                const centered = @as(i32, @intCast(i % 251)) - 125;
                value.* = @as(f32, @floatFromInt(centered)) * 0.001;
            },
            .bf16 => for (self.storage_bf16, 0..) |*value, i| {
                const centered = @as(i32, @intCast(i % 251)) - 125;
                value.* = f32ToBf16(@as(f32, @floatFromInt(centered)) * 0.001);
            },
        }
    }

    fn freeStorage(self: *OwnedModel) void {
        switch (self.kind) {
            .f32 => if (self.storage_f32.len != 0) self.allocator.free(self.storage_f32),
            .bf16 => if (self.storage_bf16.len != 0) self.allocator.free(self.storage_bf16),
        }
        self.storage_f32 = emptyAligned(f32);
        self.storage_bf16 = emptyAligned(u16);
    }

    fn assignSlices(self: *OwnedModel) void {
        var cursor: usize = 0;
        const config = self.config;
        const hidden = config.hidden_size;
        const kv = config.kvDim();
        const inter = config.intermediate_size;

        const token_embedding = self.takeMatrix(&cursor, config.vocab_size, hidden);
        for (self.layers) |*layer| {
            layer.* = .{
                .input_norm = self.takeWeight(&cursor, hidden),
                .q_proj = self.takeMatrix(&cursor, hidden, hidden),
                .q_bias = if (config.qkv_bias) self.takeWeight(&cursor, hidden) else null,
                .k_proj = self.takeMatrix(&cursor, kv, hidden),
                .k_bias = if (config.qkv_bias) self.takeWeight(&cursor, kv) else null,
                .v_proj = self.takeMatrix(&cursor, kv, hidden),
                .v_bias = if (config.qkv_bias) self.takeWeight(&cursor, kv) else null,
                .o_proj = self.takeMatrix(&cursor, hidden, hidden),
                .post_attention_norm = self.takeWeight(&cursor, hidden),
                .gate_proj = self.takeMatrix(&cursor, inter, hidden),
                .up_proj = self.takeMatrix(&cursor, inter, hidden),
                .down_proj = self.takeMatrix(&cursor, hidden, inter),
            };
        }

        const final_norm = self.takeWeight(&cursor, hidden);
        const lm_head = if (config.tie_word_embeddings) null else self.takeMatrix(&cursor, config.vocab_size, hidden);
        self.weights = .{
            .config = config,
            .token_embedding = token_embedding,
            .layers = self.layers,
            .final_norm = final_norm,
            .lm_head = lm_head,
        };
    }

    fn takeWeight(self: *OwnedModel, cursor: *usize, len: usize) WeightSlice {
        alignStorageCursor(self.kind, cursor);
        const start = cursor.*;
        cursor.* += len;
        return switch (self.kind) {
            .f32 => .{ .f32 = self.storage_f32[start..][0..len] },
            .bf16 => .{ .bf16 = self.storage_bf16[start..][0..len] },
        };
    }

    fn takeMatrix(self: *OwnedModel, cursor: *usize, rows: usize, cols: usize) Matrix {
        return .{ .rows = rows, .cols = cols, .values = self.takeWeight(cursor, rows * cols) };
    }
};

pub const KVCache = struct {
    allocator: Allocator,
    config: Config,
    max_seq_len: usize,
    keys: []align(data_alignment_bytes) f32,
    values: []align(data_alignment_bytes) f32,

    pub fn init(allocator: Allocator, config: Config, max_seq_len: usize) !KVCache {
        try config.validate();
        if (max_seq_len == 0 or max_seq_len > config.max_position_embeddings) return Error.InvalidConfig;

        const len = config.num_hidden_layers * max_seq_len * config.kvDim();
        const keys = try allocator.alignedAlloc(f32, data_alignment, len);
        errdefer allocator.free(keys);
        const values = try allocator.alignedAlloc(f32, data_alignment, len);

        return .{
            .allocator = allocator,
            .config = config,
            .max_seq_len = max_seq_len,
            .keys = keys,
            .values = values,
        };
    }

    pub fn deinit(self: *KVCache) void {
        if (self.keys.len != 0) self.allocator.free(self.keys);
        if (self.values.len != 0) self.allocator.free(self.values);
        self.* = .{
            .allocator = self.allocator,
            .config = self.config,
            .max_seq_len = 0,
            .keys = emptyAligned(f32),
            .values = emptyAligned(f32),
        };
    }

    pub fn clear(self: *KVCache) void {
        @memset(self.keys, 0.0);
        @memset(self.values, 0.0);
    }

    pub fn keySlice(self: *KVCache, layer_index: usize, position: usize) []f32 {
        const start = self.offset(layer_index, position);
        return self.keys[start..][0..self.config.kvDim()];
    }

    pub fn valueSlice(self: *KVCache, layer_index: usize, position: usize) []f32 {
        const start = self.offset(layer_index, position);
        return self.values[start..][0..self.config.kvDim()];
    }

    pub fn constKeyHead(self: *const KVCache, layer_index: usize, position: usize, kv_head: usize) []const f32 {
        const head_dim = self.config.headDim();
        const start = self.offset(layer_index, position) + kv_head * head_dim;
        return self.keys[start..][0..head_dim];
    }

    pub fn constValueHead(self: *const KVCache, layer_index: usize, position: usize, kv_head: usize) []const f32 {
        const head_dim = self.config.headDim();
        const start = self.offset(layer_index, position) + kv_head * head_dim;
        return self.values[start..][0..head_dim];
    }

    fn offset(self: *const KVCache, layer_index: usize, position: usize) usize {
        return (layer_index * self.max_seq_len + position) * self.config.kvDim();
    }
};

pub const Scratch = struct {
    allocator: Allocator,
    config: Config,
    max_seq_len: usize,
    x: []align(data_alignment_bytes) f32,
    residual: []align(data_alignment_bytes) f32,
    norm: []align(data_alignment_bytes) f32,
    q: []align(data_alignment_bytes) f32,
    k: []align(data_alignment_bytes) f32,
    v: []align(data_alignment_bytes) f32,
    attention: []align(data_alignment_bytes) f32,
    projected: []align(data_alignment_bytes) f32,
    gate: []align(data_alignment_bytes) f32,
    up: []align(data_alignment_bytes) f32,
    ff: []align(data_alignment_bytes) f32,
    scores: []align(data_alignment_bytes) f32,
    logits: []align(data_alignment_bytes) f32,

    pub fn init(allocator: Allocator, config: Config, max_seq_len: usize) !Scratch {
        try config.validate();
        if (max_seq_len == 0 or max_seq_len > config.max_position_embeddings) return Error.InvalidConfig;

        const hidden = config.hidden_size;
        const inter = config.intermediate_size;
        const kv = config.kvDim();

        var scratch = Scratch{
            .allocator = allocator,
            .config = config,
            .max_seq_len = max_seq_len,
            .x = emptyAligned(f32),
            .residual = emptyAligned(f32),
            .norm = emptyAligned(f32),
            .q = emptyAligned(f32),
            .k = emptyAligned(f32),
            .v = emptyAligned(f32),
            .attention = emptyAligned(f32),
            .projected = emptyAligned(f32),
            .gate = emptyAligned(f32),
            .up = emptyAligned(f32),
            .ff = emptyAligned(f32),
            .scores = emptyAligned(f32),
            .logits = emptyAligned(f32),
        };
        errdefer scratch.deinit();

        scratch.x = try allocator.alignedAlloc(f32, data_alignment, hidden);
        scratch.residual = try allocator.alignedAlloc(f32, data_alignment, hidden);
        scratch.norm = try allocator.alignedAlloc(f32, data_alignment, hidden);
        scratch.q = try allocator.alignedAlloc(f32, data_alignment, hidden);
        scratch.k = try allocator.alignedAlloc(f32, data_alignment, kv);
        scratch.v = try allocator.alignedAlloc(f32, data_alignment, kv);
        scratch.attention = try allocator.alignedAlloc(f32, data_alignment, hidden);
        scratch.projected = try allocator.alignedAlloc(f32, data_alignment, hidden);
        scratch.gate = try allocator.alignedAlloc(f32, data_alignment, inter);
        scratch.up = try allocator.alignedAlloc(f32, data_alignment, inter);
        scratch.ff = try allocator.alignedAlloc(f32, data_alignment, inter);
        scratch.scores = try allocator.alignedAlloc(f32, data_alignment, max_seq_len);
        scratch.logits = try allocator.alignedAlloc(f32, data_alignment, config.vocab_size);
        return scratch;
    }

    pub fn deinit(self: *Scratch) void {
        if (self.x.len != 0) self.allocator.free(self.x);
        if (self.residual.len != 0) self.allocator.free(self.residual);
        if (self.norm.len != 0) self.allocator.free(self.norm);
        if (self.q.len != 0) self.allocator.free(self.q);
        if (self.k.len != 0) self.allocator.free(self.k);
        if (self.v.len != 0) self.allocator.free(self.v);
        if (self.attention.len != 0) self.allocator.free(self.attention);
        if (self.projected.len != 0) self.allocator.free(self.projected);
        if (self.gate.len != 0) self.allocator.free(self.gate);
        if (self.up.len != 0) self.allocator.free(self.up);
        if (self.ff.len != 0) self.allocator.free(self.ff);
        if (self.scores.len != 0) self.allocator.free(self.scores);
        if (self.logits.len != 0) self.allocator.free(self.logits);
        self.* = undefined;
    }
};

pub const Runner = struct {
    weights: *const ModelWeights,
    cache: *KVCache,
    scratch: *Scratch,
    options: RuntimeOptions = .{},

    pub fn init(weights: *const ModelWeights, cache: *KVCache, scratch: *Scratch, options: RuntimeOptions) Error!Runner {
        try weights.validate();
        if (!std.meta.eql(weights.config, cache.config)) return Error.InvalidConfig;
        if (!std.meta.eql(weights.config, scratch.config)) return Error.InvalidConfig;
        if (cache.max_seq_len != scratch.max_seq_len) return Error.InvalidConfig;
        return .{ .weights = weights, .cache = cache, .scratch = scratch, .options = options };
    }

    pub fn forwardToken(self: *Runner, token_id: u32, position: usize) ![]const f32 {
        const config = self.weights.config;
        if (token_id >= config.vocab_size) return Error.InvalidToken;
        if (position >= self.cache.max_seq_len) return Error.ContextFull;

        copyEmbeddingToken(self.scratch.x, self.weights.token_embedding, token_id);
        return self.forwardCurrentEmbedding(position);
    }

    pub fn forwardEmbedding(self: *Runner, embedding: []const f32, position: usize) ![]const f32 {
        const config = self.weights.config;
        if (embedding.len != config.hidden_size) return Error.InvalidWeight;
        if (position >= self.cache.max_seq_len) return Error.ContextFull;

        @memcpy(self.scratch.x, embedding);
        return self.forwardCurrentEmbedding(position);
    }

    fn forwardCurrentEmbedding(self: *Runner, position: usize) ![]const f32 {
        const config = self.weights.config;
        for (self.weights.layers, 0..) |layer, layer_index| {
            self.prefetchAhead(layer_index);
            try self.forwardLayer(layer, layer_index, position);
        }

        rmsNormInto(self.scratch.norm, self.scratch.x, self.weights.final_norm, config.rms_norm_eps);
        try linearInto(self.scratch.logits, self.weights.logitsMatrix(), self.scratch.norm, null, self.options);
        return self.scratch.logits;
    }

    fn forwardLayer(self: *Runner, layer: LayerWeights, layer_index: usize, position: usize) !void {
        const config = self.weights.config;
        const hidden = config.hidden_size;

        @memcpy(self.scratch.residual, self.scratch.x);
        rmsNormInto(self.scratch.norm, self.scratch.x, layer.input_norm, config.rms_norm_eps);
        try linearInto(self.scratch.q, layer.q_proj, self.scratch.norm, layer.q_bias, self.options);
        try linearInto(self.scratch.k, layer.k_proj, self.scratch.norm, layer.k_bias, self.options);
        try linearInto(self.scratch.v, layer.v_proj, self.scratch.norm, layer.v_bias, self.options);

        applyRoPE(self.scratch.q, config.num_attention_heads, config.headDim(), position, config.rope_theta, config.rope_layout);
        applyRoPE(self.scratch.k, config.num_key_value_heads, config.headDim(), position, config.rope_theta, config.rope_layout);
        @memcpy(self.cache.keySlice(layer_index, position), self.scratch.k);
        @memcpy(self.cache.valueSlice(layer_index, position), self.scratch.v);

        computeAttentionInto(
            self.scratch.attention,
            self.scratch.q,
            self.cache,
            layer_index,
            position,
            config,
            self.scratch.scores,
        );
        try linearInto(self.scratch.projected, layer.o_proj, self.scratch.attention, layer.o_bias, self.options);
        addInto(self.scratch.x, self.scratch.residual, self.scratch.projected);

        @memcpy(self.scratch.residual, self.scratch.x);
        rmsNormInto(self.scratch.norm, self.scratch.x, layer.post_attention_norm, config.rms_norm_eps);
        try linearInto(self.scratch.gate, layer.gate_proj, self.scratch.norm, null, self.options);
        try linearInto(self.scratch.up, layer.up_proj, self.scratch.norm, null, self.options);
        siluMulInto(self.scratch.ff, self.scratch.gate, self.scratch.up);
        try linearInto(self.scratch.projected[0..hidden], layer.down_proj, self.scratch.ff, null, self.options);
        addInto(self.scratch.x, self.scratch.residual, self.scratch.projected[0..hidden]);
    }

    fn prefetchAhead(self: *Runner, layer_index: usize) void {
        var ahead: usize = 1;
        while (ahead <= self.options.preload_layers_ahead) : (ahead += 1) {
            const next = layer_index + ahead;
            if (next >= self.weights.layers.len) break;
            self.weights.layers[next].prefetch();
        }
    }
};

pub fn argmax(logits: []const f32) usize {
    std.debug.assert(logits.len != 0);
    var best_index: usize = 0;
    var best_value = logits[0];
    for (logits[1..], 1..) |value, i| {
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    return best_index;
}

pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const rounded = bits + 0x7fff + ((bits >> 16) & 1);
    return @intCast(rounded >> 16);
}

pub fn expandBf16Into(out: []f32, input: []const u16) Error!void {
    if (out.len != input.len) return Error.LengthMismatch;
    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        storeVec(out, i, loadBf16Vec(input, i));
    }
    while (i < out.len) : (i += 1) out[i] = bf16ToF32(input[i]);
}

pub fn storageElementCount(config: Config, kind: WeightKind) usize {
    const elem_align = elementsPerCacheLine(kind);
    var cursor: usize = 0;
    addAligned(&cursor, elem_align, config.vocab_size * config.hidden_size);
    for (0..config.num_hidden_layers) |_| {
        addAligned(&cursor, elem_align, config.hidden_size);
        addAligned(&cursor, elem_align, config.hidden_size * config.hidden_size);
        if (config.qkv_bias) addAligned(&cursor, elem_align, config.hidden_size);
        addAligned(&cursor, elem_align, config.kvDim() * config.hidden_size);
        if (config.qkv_bias) addAligned(&cursor, elem_align, config.kvDim());
        addAligned(&cursor, elem_align, config.kvDim() * config.hidden_size);
        if (config.qkv_bias) addAligned(&cursor, elem_align, config.kvDim());
        addAligned(&cursor, elem_align, config.hidden_size * config.hidden_size);
        addAligned(&cursor, elem_align, config.hidden_size);
        addAligned(&cursor, elem_align, config.intermediate_size * config.hidden_size);
        addAligned(&cursor, elem_align, config.intermediate_size * config.hidden_size);
        addAligned(&cursor, elem_align, config.hidden_size * config.intermediate_size);
    }
    addAligned(&cursor, elem_align, config.hidden_size);
    if (!config.tie_word_embeddings) {
        addAligned(&cursor, elem_align, config.vocab_size * config.hidden_size);
    }
    return cursor;
}

pub fn loadOwnedFromDirectory(allocator: Allocator, io: std.Io, model_dir: []const u8, kind: WeightKind) !OwnedModel {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    const model_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(model_path);

    const config = try Config.loadFromFile(allocator, io, config_path);
    return loadOwnedFromSafetensorsFile(allocator, io, config, kind, model_path);
}

pub fn loadOwnedFromSafetensorsFile(
    allocator: Allocator,
    io: std.Io,
    config: Config,
    kind: WeightKind,
    path: []const u8,
) !OwnedModel {
    return loadOwnedFromSafetensorsFileWithProgress(allocator, io, config, kind, path, .{});
}

pub fn loadOwnedFromSafetensorsFileWithProgress(
    allocator: Allocator,
    io: std.Io,
    config: Config,
    kind: WeightKind,
    path: []const u8,
    progress: LoadProgress,
) !OwnedModel {
    var owned = try OwnedModel.init(allocator, config, kind);
    errdefer owned.deinit();

    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    var map = try std.Io.File.MemoryMap.create(io, file, .{
        .len = @intCast(stat.size),
        .protection = .{ .read = true, .write = false },
        .populate = false,
    });
    defer map.destroy(io);

    try loadSafetensorsIntoOwnedModel(&owned, map.memory, progress);
    return owned;
}

pub fn loadOwnedFromGgufFile(
    allocator: Allocator,
    io: std.Io,
    kind: WeightKind,
    path: []const u8,
) !OwnedModel {
    return loadOwnedFromGgufFileWithProgress(allocator, io, kind, path, .{});
}

pub fn loadOwnedFromGgufFileWithProgress(
    allocator: Allocator,
    io: std.Io,
    kind: WeightKind,
    path: []const u8,
    progress: LoadProgress,
) !OwnedModel {
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

    const config = try Config.fromGguf(parsed);
    var owned = try OwnedModel.init(allocator, config, kind);
    errdefer owned.deinit();

    try loadGgufIntoOwnedModel(&owned, parsed, map.memory, progress);
    return owned;
}

fn loadSafetensorsIntoOwnedModel(owned: *OwnedModel, bytes: []const u8, progress: LoadProgress) !void {
    if (bytes.len < 8) return Error.InvalidWeight;
    const header_len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    if (8 + header_len > bytes.len) return Error.InvalidWeight;

    const header = bytes[8 .. 8 + header_len];
    const data = bytes[8 + header_len ..];
    var parsed = try std.json.parseFromSlice(std.json.Value, owned.allocator, header, .{ .parse_numbers = true });
    defer parsed.deinit();
    if (parsed.value != .object) return Error.InvalidWeight;
    const tensors = parsed.value.object;

    const config = owned.config;
    const total_steps = weightLoadStepCount(owned);
    var completed_steps: usize = 0;
    progress.report(completed_steps, total_steps, "start");

    try copyTensorIntoWeight(owned.weights.token_embedding.values, tensors, data, "model.embed_tokens.weight");
    advanceLoadProgress(progress, &completed_steps, total_steps, "model.embed_tokens.weight");

    var name_buf: [160]u8 = undefined;
    for (owned.layers, 0..) |layer, i| {
        var name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.input_layernorm.weight", .{i});
        try copyTensorIntoWeight(layer.input_norm, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_proj.weight", .{i});
        try copyTensorIntoWeight(layer.q_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        if (config.qkv_bias) {
            name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_proj.bias", .{i});
            try copyTensorIntoWeight(layer.q_bias.?, tensors, data, name);
            advanceLoadProgress(progress, &completed_steps, total_steps, name);
        }

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.k_proj.weight", .{i});
        try copyTensorIntoWeight(layer.k_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        if (config.qkv_bias) {
            name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.k_proj.bias", .{i});
            try copyTensorIntoWeight(layer.k_bias.?, tensors, data, name);
            advanceLoadProgress(progress, &completed_steps, total_steps, name);
        }

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.v_proj.weight", .{i});
        try copyTensorIntoWeight(layer.v_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        if (config.qkv_bias) {
            name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.v_proj.bias", .{i});
            try copyTensorIntoWeight(layer.v_bias.?, tensors, data, name);
            advanceLoadProgress(progress, &completed_steps, total_steps, name);
        }

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.o_proj.weight", .{i});
        try copyTensorIntoWeight(layer.o_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.post_attention_layernorm.weight", .{i});
        try copyTensorIntoWeight(layer.post_attention_norm, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.gate_proj.weight", .{i});
        try copyTensorIntoWeight(layer.gate_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.up_proj.weight", .{i});
        try copyTensorIntoWeight(layer.up_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.down_proj.weight", .{i});
        try copyTensorIntoWeight(layer.down_proj.values, tensors, data, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);
    }

    try copyTensorIntoWeight(owned.weights.final_norm, tensors, data, "model.norm.weight");
    advanceLoadProgress(progress, &completed_steps, total_steps, "model.norm.weight");
    if (owned.weights.lm_head) |head| {
        try copyTensorIntoWeight(head.values, tensors, data, "lm_head.weight");
        advanceLoadProgress(progress, &completed_steps, total_steps, "lm_head.weight");
    }
}

fn loadGgufIntoOwnedModel(owned: *OwnedModel, file: gguf.File, bytes: []const u8, progress: LoadProgress) !void {
    const config = owned.config;
    if (!config.supportsDenseRunner()) return Error.UnsupportedModelArchitecture;
    if (config.qkv_bias) return Error.UnsupportedModelArchitecture;

    const total_steps = weightLoadStepCount(owned);
    var completed_steps: usize = 0;
    progress.report(completed_steps, total_steps, "start");

    try copyGgufTensorIntoWeight(owned.weights.token_embedding.values, file, bytes, "token_embd.weight");
    advanceLoadProgress(progress, &completed_steps, total_steps, "token_embd.weight");

    var name_buf: [96]u8 = undefined;
    for (owned.layers, 0..) |layer, i| {
        var name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{i});
        try copyGgufTensorIntoWeight(layer.input_norm, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{i});
        try copyGgufTensorIntoWeight(layer.q_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{i});
        try copyGgufTensorIntoWeight(layer.k_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{i});
        try copyGgufTensorIntoWeight(layer.v_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{i});
        try copyGgufTensorIntoWeight(layer.o_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{i});
        try copyGgufTensorIntoWeight(layer.post_attention_norm, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{i});
        try copyGgufTensorIntoWeight(layer.gate_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{i});
        try copyGgufTensorIntoWeight(layer.up_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);

        name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{i});
        try copyGgufTensorIntoWeight(layer.down_proj.values, file, bytes, name);
        advanceLoadProgress(progress, &completed_steps, total_steps, name);
    }

    try copyGgufTensorIntoWeight(owned.weights.final_norm, file, bytes, "output_norm.weight");
    advanceLoadProgress(progress, &completed_steps, total_steps, "output_norm.weight");
    if (owned.weights.lm_head) |head| {
        try copyGgufTensorIntoWeight(head.values, file, bytes, "output.weight");
        advanceLoadProgress(progress, &completed_steps, total_steps, "output.weight");
    }
}

fn weightLoadStepCount(owned: *const OwnedModel) usize {
    var per_layer: usize = 9;
    if (owned.config.qkv_bias) per_layer += 3;

    var total: usize = 1 + owned.layers.len * per_layer + 1;
    if (owned.weights.lm_head != null) total += 1;
    return total;
}

fn advanceLoadProgress(progress: LoadProgress, completed: *usize, total: usize, name: []const u8) void {
    completed.* += 1;
    progress.report(completed.*, total, name);
}

fn copyGgufTensorIntoWeight(weight: WeightSlice, file: gguf.File, bytes: []const u8, name: []const u8) !void {
    const info = file.tensor(name) orelse return Error.MissingSafetensorsTensor;
    if (!info.tensor_type.isNativeFloat()) return Error.UnsupportedDType;
    if (info.elementCount() != weight.len()) return Error.InvalidWeight;

    const byte_len = try ggufTensorByteLen(info);
    const start = file.data_start + @as(usize, @intCast(info.offset));
    const end = start + byte_len;
    if (end > bytes.len or end < start) return Error.InvalidWeight;
    const raw = bytes[start..end];

    switch (weight) {
        .f32 => |dst| switch (info.tensor_type) {
            .f32 => if (raw.len == dst.len * @sizeOf(f32)) {
                @memcpy(std.mem.sliceAsBytes(@constCast(dst)), raw);
            } else return Error.InvalidWeight,
            .bf16 => {
                if (raw.len != dst.len * @sizeOf(u16)) return Error.InvalidWeight;
                var i: usize = 0;
                while (i < dst.len) : (i += 1) {
                    const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                    @constCast(dst)[i] = bf16ToF32(bits);
                }
            },
            .f16 => {
                if (raw.len != dst.len * @sizeOf(u16)) return Error.InvalidWeight;
                var i: usize = 0;
                while (i < dst.len) : (i += 1) {
                    const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                    @constCast(dst)[i] = f16ToF32(bits);
                }
            },
            else => return Error.UnsupportedDType,
        },
        .bf16 => |dst| switch (info.tensor_type) {
            .bf16 => if (raw.len == dst.len * @sizeOf(u16)) {
                @memcpy(std.mem.sliceAsBytes(@constCast(dst)), raw);
            } else return Error.InvalidWeight,
            .f16 => {
                if (raw.len != dst.len * @sizeOf(u16)) return Error.InvalidWeight;
                var i: usize = 0;
                while (i < dst.len) : (i += 1) {
                    const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                    @constCast(dst)[i] = f32ToBf16(f16ToF32(bits));
                }
            },
            .f32 => {
                if (raw.len != dst.len * @sizeOf(f32)) return Error.InvalidWeight;
                var i: usize = 0;
                while (i < dst.len) : (i += 1) {
                    const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
                    @constCast(dst)[i] = f32ToBf16(@bitCast(bits));
                }
            },
            else => return Error.UnsupportedDType,
        },
    }
}

fn ggufTensorByteLen(info: gguf.TensorInfo) !usize {
    const elem_size: usize = switch (info.tensor_type) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        else => return Error.UnsupportedDType,
    };
    return info.elementCount() * elem_size;
}

fn copyTensorIntoWeight(weight: WeightSlice, tensors: std.json.ObjectMap, data: []const u8, name: []const u8) !void {
    const meta = tensors.get(name) orelse return Error.MissingSafetensorsTensor;
    const info = try parseTensorInfo(meta);
    if (shapeElementCount(info.shape) != weight.len()) return Error.InvalidWeight;
    if (info.end > data.len or info.end < info.start) return Error.InvalidWeight;
    const raw = data[info.start..info.end];

    switch (weight) {
        .f32 => |dst| switch (info.dtype) {
            .f32 => if (raw.len == dst.len * @sizeOf(f32)) {
                @memcpy(std.mem.sliceAsBytes(@constCast(dst)), raw);
            } else return Error.InvalidWeight,
            .bf16 => {
                if (raw.len != dst.len * @sizeOf(u16)) return Error.InvalidWeight;
                var i: usize = 0;
                while (i < dst.len) : (i += 1) {
                    const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                    @constCast(dst)[i] = bf16ToF32(bits);
                }
            },
        },
        .bf16 => |dst| switch (info.dtype) {
            .bf16 => if (raw.len == dst.len * @sizeOf(u16)) {
                @memcpy(std.mem.sliceAsBytes(@constCast(dst)), raw);
            } else return Error.InvalidWeight,
            .f32 => {
                if (raw.len != dst.len * @sizeOf(f32)) return Error.InvalidWeight;
                var i: usize = 0;
                while (i < dst.len) : (i += 1) {
                    const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
                    @constCast(dst)[i] = f32ToBf16(@bitCast(bits));
                }
            },
        },
    }
}

const TensorDType = enum { f32, bf16 };

const TensorInfo = struct {
    dtype: TensorDType,
    shape: []const std.json.Value,
    start: usize,
    end: usize,
};

fn parseTensorInfo(value: std.json.Value) !TensorInfo {
    if (value != .object) return Error.InvalidWeight;
    const object = value.object;
    const dtype_text = getString(object, "dtype") orelse return Error.InvalidWeight;
    const dtype: TensorDType = if (std.mem.eql(u8, dtype_text, "F32"))
        .f32
    else if (std.mem.eql(u8, dtype_text, "BF16"))
        .bf16
    else
        return Error.UnsupportedDType;

    const shape_value = object.get("shape") orelse return Error.InvalidWeight;
    if (shape_value != .array) return Error.InvalidWeight;

    const offsets_value = object.get("data_offsets") orelse return Error.InvalidWeight;
    if (offsets_value != .array or offsets_value.array.items.len != 2) return Error.InvalidWeight;

    return .{
        .dtype = dtype,
        .shape = shape_value.array.items,
        .start = try jsonValueUsize(offsets_value.array.items[0]),
        .end = try jsonValueUsize(offsets_value.array.items[1]),
    };
}

fn shapeElementCount(shape: []const std.json.Value) usize {
    var total: usize = 1;
    for (shape) |dim| {
        total *= jsonValueUsize(dim) catch return 0;
    }
    return total;
}

fn linearInto(out: []f32, matrix: Matrix, input: []const f32, bias: ?WeightSlice, options: RuntimeOptions) !void {
    try expectMatrix(matrix, out.len, input.len);
    if (bias) |b| try expectWeightLen(b, out.len);

    const workers = resolveWorkerCount(out.len, options);
    if (workers == 1) {
        linearRange(out, matrix, input, bias, 0, out.len);
        return;
    }

    var threads: [max_worker_threads]std.Thread = undefined;
    var jobs: [max_worker_threads]LinearJob = undefined;
    const rows_per_worker = (out.len + workers - 1) / workers;

    var spawned: usize = 0;
    while (spawned + 1 < workers) : (spawned += 1) {
        const start = spawned * rows_per_worker;
        const end = @min(start + rows_per_worker, out.len);
        jobs[spawned] = .{ .out = out, .matrix = matrix, .input = input, .bias = bias, .start = start, .end = end };
        threads[spawned] = std.Thread.spawn(.{}, linearWorker, .{&jobs[spawned]}) catch {
            for (threads[0..spawned]) |thread| thread.join();
            return Error.ThreadSpawnFailed;
        };
    }

    const start = spawned * rows_per_worker;
    linearRange(out, matrix, input, bias, start, out.len);
    for (threads[0..spawned]) |thread| thread.join();
}

const LinearJob = struct {
    out: []f32,
    matrix: Matrix,
    input: []const f32,
    bias: ?WeightSlice,
    start: usize,
    end: usize,
};

fn linearWorker(job: *const LinearJob) void {
    linearRange(job.out, job.matrix, job.input, job.bias, job.start, job.end);
}

fn linearRange(out: []f32, matrix: Matrix, input: []const f32, bias: ?WeightSlice, start: usize, end: usize) void {
    for (start..end) |row| {
        out[row] = matrix.dotRow(row, input) + if (bias) |b| b.at(row) else 0.0;
    }
}

fn resolveWorkerCount(row_count: usize, options: RuntimeOptions) usize {
    if (row_count == 0) return 1;
    const requested = if (options.thread_count == 0) std.Thread.getCpuCount() catch 1 else options.thread_count;
    if (requested <= 1) return 1;

    const min_rows = @max(options.min_rows_per_thread, 1);
    const useful = @max(@as(usize, 1), row_count / min_rows);
    return @max(@as(usize, 1), @min(@min(requested, useful), max_worker_threads));
}

fn rmsNormInto(out: []f32, input: []const f32, weight: WeightSlice, epsilon: f32) void {
    std.debug.assert(out.len == input.len);
    std.debug.assert(weight.len() == input.len);

    const mean_square = sumSquares(input) / @as(f32, @floatFromInt(input.len));
    const scale = 1.0 / @sqrt(mean_square + epsilon);
    const scale_v: Vec = @splat(scale);

    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        storeVec(out, i, loadVec(input, i) * loadWeightVec(weight, i) * scale_v);
    }
    while (i < out.len) : (i += 1) out[i] = input[i] * weight.at(i) * scale;
}

fn computeAttentionInto(
    out: []f32,
    q: []const f32,
    cache: *const KVCache,
    layer_index: usize,
    position: usize,
    config: Config,
    scores: []f32,
) void {
    const head_dim = config.headDim();
    const group = config.kvGroupSize();
    const inv_sqrt_head = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(out, 0.0);

    for (0..config.num_attention_heads) |q_head| {
        const kv_head = q_head / group;
        const q_head_slice = q[q_head * head_dim ..][0..head_dim];
        const used_scores = scores[0 .. position + 1];

        for (used_scores, 0..) |*score, t| {
            const key = cache.constKeyHead(layer_index, t, kv_head);
            score.* = dotF32(q_head_slice, key) * inv_sqrt_head;
        }
        softmaxInPlace(used_scores);

        const out_head = out[q_head * head_dim ..][0..head_dim];
        @memset(out_head, 0.0);
        for (used_scores, 0..) |score, t| {
            const value = cache.constValueHead(layer_index, t, kv_head);
            addScaled(out_head, value, score);
        }
    }
}

fn applyRoPE(values: []f32, heads: usize, head_dim: usize, position: usize, theta: f32, layout: RopeLayout) void {
    std.debug.assert(values.len == heads * head_dim);

    const pos_f = @as(f32, @floatFromInt(position));
    const dim_f = @as(f32, @floatFromInt(head_dim));
    for (0..heads) |head| {
        const head_slice = values[head * head_dim ..][0..head_dim];
        switch (layout) {
            .interleaved => {
                var i: usize = 0;
                while (i < head_dim) : (i += 2) {
                    rotatePair(head_slice, i, i + 1, i, dim_f, pos_f, theta);
                }
            },
            .split_half => {
                const half = head_dim / 2;
                for (0..half) |i| {
                    rotatePair(head_slice, i, i + half, i * 2, dim_f, pos_f, theta);
                }
            },
        }
    }
}

fn rotatePair(values: []f32, first: usize, second: usize, dim_index: usize, dim_f: f32, pos_f: f32, theta: f32) void {
    const exponent = @as(f32, @floatFromInt(dim_index)) / dim_f;
    const inv_freq = 1.0 / std.math.pow(f32, theta, exponent);
    const angle = pos_f * inv_freq;
    const cos_v = std.math.cos(angle);
    const sin_v = std.math.sin(angle);
    const x0 = values[first];
    const x1 = values[second];
    values[first] = x0 * cos_v - x1 * sin_v;
    values[second] = x0 * sin_v + x1 * cos_v;
}

fn copyEmbeddingToken(out: []f32, embedding: Matrix, token_id: u32) void {
    const token_index: usize = @intCast(token_id);
    const start = token_index * embedding.cols;
    switch (embedding.values) {
        .f32 => |values| @memcpy(out, values[start..][0..embedding.cols]),
        .bf16 => |values| expandBf16Into(out, values[start..][0..embedding.cols]) catch unreachable,
    }
}

fn siluMulInto(out: []f32, gate: []const f32, up: []const f32) void {
    std.debug.assert(out.len == gate.len and out.len == up.len);
    for (out, gate, up) |*dst, g, u| {
        dst.* = (g / (1.0 + std.math.exp(-g))) * u;
    }
}

fn addInto(out: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(out.len == a.len and out.len == b.len);
    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        storeVec(out, i, loadVec(a, i) + loadVec(b, i));
    }
    while (i < out.len) : (i += 1) out[i] = a[i] + b[i];
}

fn addScaled(out: []f32, values: []const f32, scale: f32) void {
    std.debug.assert(out.len == values.len);
    const scale_v: Vec = @splat(scale);
    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        storeVec(out, i, loadVec(out, i) + loadVec(values, i) * scale_v);
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

fn sumSquares(values: []const f32) f32 {
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + simd_lanes <= values.len) : (i += simd_lanes) {
        const v = loadVec(values, i);
        acc += v * v;
    }
    var sum = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) sum = @mulAdd(f32, values[i], values[i], sum);
    return sum;
}

fn dotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + simd_lanes <= a.len) : (i += simd_lanes) {
        acc += loadVec(a, i) * loadVec(b, i);
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum = @mulAdd(f32, a[i], b[i], sum);
    return sum;
}

fn dotBf16(a: []const u16, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + simd_lanes <= a.len) : (i += simd_lanes) {
        acc += loadBf16Vec(a, i) * loadVec(b, i);
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum = @mulAdd(f32, bf16ToF32(a[i]), b[i], sum);
    return sum;
}

inline fn loadVec(slice: []const f32, index: usize) Vec {
    return @as(Vec, slice[index..][0..simd_lanes].*);
}

inline fn loadBf16Vec(slice: []const u16, index: usize) Vec {
    const bf16 = @as(U16Vec, slice[index..][0..simd_lanes].*);
    const wide: U32Vec = @intCast(bf16);
    const bits = wide << @as(U32Vec, @splat(16));
    return @bitCast(bits);
}

inline fn loadWeightVec(weight: WeightSlice, index: usize) Vec {
    return switch (weight) {
        .f32 => |values| loadVec(values, index),
        .bf16 => |values| loadBf16Vec(values, index),
    };
}

inline fn storeVec(slice: []f32, index: usize, vec: Vec) void {
    inline for (0..simd_lanes) |lane| {
        slice[index + lane] = vec[lane];
    }
}

fn expectWeightLen(weight: WeightSlice, expected: usize) Error!void {
    if (weight.len() != expected) return Error.InvalidWeight;
}

fn expectMatrix(matrix: Matrix, rows: usize, cols: usize) Error!void {
    try matrix.validate();
    if (matrix.rows != rows or matrix.cols != cols) return Error.InvalidWeight;
}

fn addAligned(cursor: *usize, elem_align: usize, len: usize) void {
    cursor.* = std.mem.alignForward(usize, cursor.*, elem_align);
    cursor.* += len;
}

fn alignStorageCursor(kind: WeightKind, cursor: *usize) void {
    cursor.* = std.mem.alignForward(usize, cursor.*, elementsPerCacheLine(kind));
}

fn elementsPerCacheLine(kind: WeightKind) usize {
    return switch (kind) {
        .f32 => data_alignment_bytes / @sizeOf(f32),
        .bf16 => data_alignment_bytes / @sizeOf(u16),
    };
}

fn emptyAligned(comptime T: type) []align(data_alignment_bytes) T {
    const ptr: [*]align(data_alignment_bytes) T = @ptrFromInt(data_alignment_bytes);
    return ptr[0..0];
}

fn requiredGgufUsize(file: gguf.File, prefix: []const u8, suffix: []const u8) !usize {
    return optionalGgufUsize(file, prefix, suffix) orelse Error.MissingJsonField;
}

fn optionalGgufUsize(file: gguf.File, prefix: []const u8, suffix: []const u8) ?usize {
    var key_buf: [96]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ prefix, suffix }) catch return null;
    return file.usizeValue(key);
}

fn optionalGgufF32(file: gguf.File, prefix: []const u8, suffix: []const u8) ?f32 {
    var key_buf: [96]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ prefix, suffix }) catch return null;
    return file.f32Value(key);
}

fn getRequiredUsize(object: std.json.ObjectMap, key: []const u8) !usize {
    return jsonValueUsize(object.get(key) orelse return Error.MissingJsonField);
}

fn getOptionalUsize(object: std.json.ObjectMap, key: []const u8) ?usize {
    const value = object.get(key) orelse return null;
    return jsonValueUsize(value) catch null;
}

fn getOptionalTokenId(object: std.json.ObjectMap, key: []const u8) ?u32 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .null => null,
        .array => |array| if (array.items.len == 0) null else @intCast(jsonValueUsize(array.items[0]) catch return null),
        else => @intCast(jsonValueUsize(value) catch return null),
    };
}

fn getOptionalF32(object: std.json.ObjectMap, key: []const u8) ?f32 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        .number_string => |s| std.fmt.parseFloat(f32, s) catch null,
        else => null,
    };
}

fn getOptionalBool(object: std.json.ObjectMap, key: []const u8) ?bool {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .bool => |b| b,
        else => null,
    };
}

fn getString(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

fn jsonValueUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |i| if (i >= 0) @intCast(i) else Error.InvalidConfig,
        .number_string => |s| std.fmt.parseInt(usize, s, 10) catch Error.InvalidConfig,
        else => Error.InvalidConfig,
    };
}

test "dense qwen config parses from json instead of hardcoded model constants" {
    const config = try Config.fromJson(std.testing.allocator,
        \\{
        \\  "model_type": "qwen2",
        \\  "vocab_size": 151936,
        \\  "hidden_size": 896,
        \\  "intermediate_size": 4864,
        \\  "num_hidden_layers": 24,
        \\  "num_attention_heads": 14,
        \\  "num_key_value_heads": 2,
        \\  "max_position_embeddings": 32768,
        \\  "rope_theta": 1000000.0,
        \\  "rms_norm_eps": 1e-6,
        \\  "tie_word_embeddings": true,
        \\  "eos_token_id": 151645,
        \\  "pad_token_id": 151654
        \\}
    );
    try config.validate();

    try std.testing.expectEqual(@as(usize, 64), config.headDim());
    try std.testing.expectEqual(@as(usize, 128), config.kvDim());
    try std.testing.expectEqual(@as(usize, 7), config.kvGroupSize());
    try std.testing.expect(config.parameterCount() > 494_005_120);
    try std.testing.expect(storageElementCount(config, .bf16) >= config.parameterCount());
    try std.testing.expect(storageElementCount(config, .bf16) * @sizeOf(u16) < storageElementCount(config, .f32) * @sizeOf(f32));
}

test "dense llama config parses as a supported non-moe architecture" {
    const config = try Config.fromJson(std.testing.allocator,
        \\{
        \\  "model_type": "llama",
        \\  "vocab_size": 128256,
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 8192,
        \\  "num_hidden_layers": 16,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "max_position_embeddings": 131072,
        \\  "rope_theta": 500000.0,
        \\  "rms_norm_eps": 1e-5,
        \\  "tie_word_embeddings": true,
        \\  "attention_bias": false,
        \\  "eos_token_id": 128009,
        \\  "pad_token_id": 128004
        \\}
    );

    try config.validate();
    try std.testing.expectEqual(Architecture.llama, config.architecture);
    try std.testing.expectEqual(FeedForwardKind.dense, config.feed_forward);
    try std.testing.expect(config.supportsDenseRunner());
    try std.testing.expectEqual(@as(usize, 64), config.headDim());
}

test "dense mistral config defaults to split-half rope and no qkv bias" {
    const config = try Config.fromJson(std.testing.allocator,
        \\{
        \\  "model_type": "mistral",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 4096,
        \\  "intermediate_size": 14336,
        \\  "num_hidden_layers": 32,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 8,
        \\  "max_position_embeddings": 32768,
        \\  "rope_theta": 10000.0,
        \\  "rms_norm_eps": 1e-5,
        \\  "tie_word_embeddings": false,
        \\  "eos_token_id": 2,
        \\  "pad_token_id": 2
        \\}
    );

    try config.validate();
    try std.testing.expectEqual(Architecture.mistral, config.architecture);
    try std.testing.expectEqual(RopeLayout.split_half, config.rope_layout);
    try std.testing.expect(!config.qkv_bias);
    try std.testing.expect(config.supportsDenseRunner());
}

test "olmoe config parses as moe but is not forced through dense runner" {
    const config = try Config.fromJson(std.testing.allocator,
        \\{
        \\  "model_type": "olmoe",
        \\  "vocab_size": 50304,
        \\  "hidden_size": 2048,
        \\  "intermediate_size": 1024,
        \\  "num_hidden_layers": 16,
        \\  "num_attention_heads": 16,
        \\  "num_key_value_heads": 16,
        \\  "num_experts": 64,
        \\  "num_experts_per_tok": 8,
        \\  "max_position_embeddings": 4096,
        \\  "rope_theta": 10000.0,
        \\  "rms_norm_eps": 1e-5,
        \\  "tie_word_embeddings": false,
        \\  "attention_bias": false,
        \\  "eos_token_id": 50279,
        \\  "pad_token_id": 1
        \\}
    );

    try config.validate();
    try std.testing.expectEqual(Architecture.olmoe, config.architecture);
    try std.testing.expectEqual(FeedForwardKind.moe, config.feed_forward);
    try std.testing.expect(config.isMoe());
    try std.testing.expect(!config.supportsDenseRunner());
    try std.testing.expectError(Error.UnsupportedModelArchitecture, OwnedModel.init(std.testing.allocator, config, .bf16));
}

test "bf16 conversion and vector expansion" {
    const allocator = std.testing.allocator;
    const input = [_]u16{ f32ToBf16(1.0), f32ToBf16(-2.5), f32ToBf16(0.25), f32ToBf16(8.0) };
    const out = try allocator.alloc(f32, input.len);
    defer allocator.free(out);

    try expandBf16Into(out, &input);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), out[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), out[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out[3], 0.01);
}

test "rope layout distinguishes split-half safetensors from interleaved gguf order" {
    var split = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var interleaved = split;

    applyRoPE(&split, 1, 4, 1, 10_000.0, .split_half);
    applyRoPE(&interleaved, 1, 4, 1, 10_000.0, .interleaved);

    const cos0 = std.math.cos(@as(f32, 1.0));
    const sin0 = std.math.sin(@as(f32, 1.0));
    try std.testing.expectApproxEqAbs(1.0 * cos0 - 3.0 * sin0, split[0], 1e-6);
    try std.testing.expectApproxEqAbs(1.0 * sin0 + 3.0 * cos0, split[2], 1e-6);
    try std.testing.expectApproxEqAbs(1.0 * cos0 - 2.0 * sin0, interleaved[0], 1e-6);
    try std.testing.expectApproxEqAbs(1.0 * sin0 + 2.0 * cos0, interleaved[1], 1e-6);
}

test "owned model lays weights out in one aligned contiguous blob" {
    const allocator = std.testing.allocator;
    const config: Config = .{
        .vocab_size = 32,
        .hidden_size = 16,
        .intermediate_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .max_position_embeddings = 32,
        .rope_theta = 10_000.0,
        .rms_norm_eps = 0.000001,
        .tie_word_embeddings = true,
        .eos_token_id = 2,
        .pad_token_id = 0,
    };

    var model = try OwnedModel.init(allocator, config, .bf16);
    defer model.deinit();

    try model.weights.validate();
    try std.testing.expect(data_alignment.check(@intFromPtr(model.storage_bf16.ptr)));
    try std.testing.expect(model.byteLen() >= config.parameterCount() * @sizeOf(u16));
}

test "runner executes a tiny qwen-shaped token step" {
    const allocator = std.testing.allocator;
    const config: Config = .{
        .vocab_size = 24,
        .hidden_size = 16,
        .intermediate_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .max_position_embeddings = 16,
        .rope_theta = 10_000.0,
        .rms_norm_eps = 0.000001,
        .tie_word_embeddings = true,
        .eos_token_id = 2,
        .pad_token_id = 0,
    };

    var model = try OwnedModel.init(allocator, config, .f32);
    defer model.deinit();
    model.fillDeterministicForTesting();

    var cache = try KVCache.init(allocator, config, 8);
    defer cache.deinit();
    cache.clear();

    var scratch = try Scratch.init(allocator, config, 8);
    defer scratch.deinit();

    var runner = try Runner.init(&model.weights, &cache, &scratch, .{ .thread_count = 2, .min_rows_per_thread = 4 });
    const logits = try runner.forwardToken(3, 0);

    try std.testing.expectEqual(config.vocab_size, logits.len);
    for (logits) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expect(argmax(logits) < config.vocab_size);

    var embedding: [16]f32 = @splat(0.01);
    const embedding_logits = try runner.forwardEmbedding(&embedding, 1);
    try std.testing.expectEqual(config.vocab_size, embedding_logits.len);
    for (embedding_logits) |value| try std.testing.expect(std.math.isFinite(value));
}

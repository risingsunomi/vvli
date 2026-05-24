// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const llm = @import("llm");

pub const Allocator = std.mem.Allocator;

pub const Error = llm.Error || error{
    EmptyPrompt,
    OutputTooSmall,
    InvalidSamplingOptions,
};

pub const Options = struct {
    max_new_tokens: usize = 32,
    eos_token_id: ?u32 = null,
    temperature: f32 = 0.8,
    top_p: f32 = 0.95,
    top_k: usize = 40,
    repeat_penalty: f32 = 1.10,
    repeat_last_n: usize = 64,
    seed: ?u64 = null,
};

pub const Stats = struct {
    prompt_tokens: usize,
    generated_tokens: usize,
    prefill_nanoseconds: i96,
    decode_nanoseconds: i96,
    total_nanoseconds: i96,

    pub fn decodeSeconds(self: Stats) f64 {
        return secondsFromNanoseconds(self.decode_nanoseconds);
    }

    pub fn decodeTokensPerSecond(self: Stats) f64 {
        return tokensPerSecond(self.generated_tokens, self.decode_nanoseconds);
    }

    pub fn totalTokensPerSecond(self: Stats) f64 {
        return tokensPerSecond(self.prompt_tokens + self.generated_tokens, self.total_nanoseconds);
    }
};

pub const Result = struct {
    tokens: []u32,
    stats: Stats,
};

pub const TokenSink = struct {
    context: ?*anyopaque = null,
    on_token: *const fn (context: ?*anyopaque, token: u32, index: usize) anyerror!void,

    fn emit(self: TokenSink, token: u32, index: usize) !void {
        try self.on_token(self.context, token, index);
    }
};

const Candidate = struct {
    id: u32,
    logit: f32,
    probability: f32 = 0.0,
};

const Sampler = struct {
    allocator: Allocator,
    candidates: []Candidate,
    repeat_counts: []u16,
    prng: std.Random.DefaultPrng,

    fn init(allocator: Allocator, io: std.Io, vocab_size: usize, options: Options) !Sampler {
        var seed = options.seed orelse seed: {
            var seed_bytes: [8]u8 = undefined;
            io.random(&seed_bytes);
            break :seed std.mem.readInt(u64, &seed_bytes, .little);
        };
        if (seed == 0) seed = 0x9e3779b97f4a7c15;

        const candidates = try allocator.alloc(Candidate, vocab_size);
        errdefer allocator.free(candidates);
        const repeat_counts = try allocator.alloc(u16, vocab_size);

        return .{
            .allocator = allocator,
            .candidates = candidates,
            .repeat_counts = repeat_counts,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    fn deinit(self: *Sampler) void {
        self.allocator.free(self.candidates);
        self.allocator.free(self.repeat_counts);
        self.* = undefined;
    }

    fn next(
        self: *Sampler,
        logits: []const f32,
        prompt_tokens: []const u32,
        generated_tokens: []const u32,
        options: Options,
    ) !u32 {
        if (logits.len > self.candidates.len or logits.len > self.repeat_counts.len) return Error.OutputTooSmall;
        try validateSamplingOptions(options);
        buildRepeatCounts(self.repeat_counts[0..logits.len], prompt_tokens, generated_tokens, options);

        if (options.temperature <= 0.0) {
            return greedyWithPenalties(logits, self.repeat_counts[0..logits.len], options);
        }

        const candidate_count = buildCandidates(
            self.candidates[0..logits.len],
            logits,
            self.repeat_counts[0..logits.len],
            options,
        );
        if (candidate_count == 0) return @intCast(llm.argmax(logits));

        const candidates = self.candidates[0..candidate_count];
        const sample_count = prepareProbabilities(candidates, options);
        if (sample_count == 0) return candidates[0].id;

        var random = self.prng.random();
        return sampleCandidate(candidates[0..sample_count], random.float(f32));
    }
};

pub const EmbeddingRange = struct {
    start: usize,
    token_count: usize,
    dimensions: usize,
    data: []const f32,

    pub fn token(self: EmbeddingRange, index: usize) ?[]const f32 {
        if (index < self.start or index >= self.start + self.token_count) return null;
        const offset = (index - self.start) * self.dimensions;
        return self.data[offset..][0..self.dimensions];
    }
};

pub fn generateGreedy(
    allocator: Allocator,
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    options: Options,
) ![]u32 {
    if (prompt_tokens.len == 0) return Error.EmptyPrompt;

    const out = try allocator.alloc(u32, options.max_new_tokens);
    errdefer allocator.free(out);

    var position: usize = 0;
    var logits: []const f32 = &.{};
    for (prompt_tokens) |token| {
        logits = try runner.forwardToken(token, position);
        position += 1;
    }

    var produced: usize = 0;
    while (produced < options.max_new_tokens) : (produced += 1) {
        const next: u32 = @intCast(llm.argmax(logits));
        out[produced] = next;
        if (options.eos_token_id) |eos| {
            if (next == eos) {
                produced += 1;
                break;
            }
        }
        logits = try runner.forwardToken(next, position);
        position += 1;
    }

    return allocator.realloc(out, produced);
}

pub fn generateGreedyMeasured(
    allocator: Allocator,
    io: std.Io,
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    options: Options,
) !Result {
    return generateGreedyMeasuredInternal(allocator, io, runner, prompt_tokens, null, options, null);
}

pub fn generateGreedyMeasuredStreaming(
    allocator: Allocator,
    io: std.Io,
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    options: Options,
    sink: TokenSink,
) !Result {
    return generateGreedyMeasuredInternal(allocator, io, runner, prompt_tokens, null, options, sink);
}

pub fn generateGreedyMeasuredWithEmbeddings(
    allocator: Allocator,
    io: std.Io,
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    embeddings: EmbeddingRange,
    options: Options,
) !Result {
    return generateGreedyMeasuredInternal(allocator, io, runner, prompt_tokens, embeddings, options, null);
}

pub fn generateGreedyMeasuredWithEmbeddingsStreaming(
    allocator: Allocator,
    io: std.Io,
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    embeddings: EmbeddingRange,
    options: Options,
    sink: TokenSink,
) !Result {
    return generateGreedyMeasuredInternal(allocator, io, runner, prompt_tokens, embeddings, options, sink);
}

fn generateGreedyMeasuredInternal(
    allocator: Allocator,
    io: std.Io,
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    embeddings: ?EmbeddingRange,
    options: Options,
    sink: ?TokenSink,
) !Result {
    if (prompt_tokens.len == 0) return Error.EmptyPrompt;
    if (embeddings) |range| {
        if (range.start + range.token_count > prompt_tokens.len) return Error.OutputTooSmall;
    }
    try validateSamplingOptions(options);

    const out = try allocator.alloc(u32, options.max_new_tokens);
    errdefer allocator.free(out);

    const total_start = std.Io.Clock.awake.now(io);

    var position: usize = 0;
    var logits: []const f32 = &.{};
    for (prompt_tokens, 0..) |token, index| {
        const maybe_embedding = if (embeddings) |range| range.token(index) else null;
        if (maybe_embedding) |embedding| {
            logits = try runner.forwardEmbedding(embedding, position);
        } else {
            logits = try runner.forwardToken(token, position);
        }
        position += 1;
    }

    const prefill_done = std.Io.Clock.awake.now(io);
    var sampler = try Sampler.init(allocator, io, logits.len, options);
    defer sampler.deinit();

    var stream_nanoseconds: i96 = 0;
    var produced: usize = 0;
    while (produced < options.max_new_tokens) {
        const next = try sampler.next(logits, prompt_tokens, out[0..produced], options);
        out[produced] = next;
        if (sink) |token_sink| {
            const stream_start = std.Io.Clock.awake.now(io);
            try token_sink.emit(next, produced);
            const stream_done = std.Io.Clock.awake.now(io);
            stream_nanoseconds += stream_start.durationTo(stream_done).toNanoseconds();
        }
        produced += 1;
        if (options.eos_token_id) |eos| {
            if (next == eos) break;
        }
        logits = try runner.forwardToken(next, position);
        position += 1;
    }

    const decode_done = std.Io.Clock.awake.now(io);
    const raw_decode_nanoseconds = prefill_done.durationTo(decode_done).toNanoseconds();
    const model_decode_nanoseconds = if (raw_decode_nanoseconds > stream_nanoseconds)
        raw_decode_nanoseconds - stream_nanoseconds
    else
        0;
    const tokens = try allocator.realloc(out, produced);
    return .{
        .tokens = tokens,
        .stats = .{
            .prompt_tokens = prompt_tokens.len,
            .generated_tokens = produced,
            .prefill_nanoseconds = total_start.durationTo(prefill_done).toNanoseconds(),
            .decode_nanoseconds = model_decode_nanoseconds,
            .total_nanoseconds = total_start.durationTo(decode_done).toNanoseconds(),
        },
    };
}

pub fn generateGreedyInto(
    runner: *llm.Runner,
    prompt_tokens: []const u32,
    out: []u32,
    options: Options,
) !usize {
    if (prompt_tokens.len == 0) return Error.EmptyPrompt;
    if (out.len < options.max_new_tokens) return Error.OutputTooSmall;

    var position: usize = 0;
    var logits: []const f32 = &.{};
    for (prompt_tokens) |token| {
        logits = try runner.forwardToken(token, position);
        position += 1;
    }

    var produced: usize = 0;
    while (produced < options.max_new_tokens) : (produced += 1) {
        const next: u32 = @intCast(llm.argmax(logits));
        out[produced] = next;
        if (options.eos_token_id) |eos| {
            if (next == eos) {
                produced += 1;
                break;
            }
        }
        logits = try runner.forwardToken(next, position);
        position += 1;
    }
    return produced;
}

fn tokensPerSecond(tokens: usize, nanoseconds: i96) f64 {
    if (tokens == 0 or nanoseconds <= 0) return 0;
    return (@as(f64, @floatFromInt(tokens)) * @as(f64, std.time.ns_per_s)) /
        @as(f64, @floatFromInt(nanoseconds));
}

fn secondsFromNanoseconds(nanoseconds: i96) f64 {
    if (nanoseconds <= 0) return 0;
    return @as(f64, @floatFromInt(nanoseconds)) / @as(f64, std.time.ns_per_s);
}

fn validateSamplingOptions(options: Options) Error!void {
    if (options.temperature < 0.0 or std.math.isNan(options.temperature)) return Error.InvalidSamplingOptions;
    if (options.top_p <= 0.0 or options.top_p > 1.0 or std.math.isNan(options.top_p)) return Error.InvalidSamplingOptions;
    if (options.repeat_penalty < 1.0 or std.math.isNan(options.repeat_penalty)) return Error.InvalidSamplingOptions;
}

fn buildRepeatCounts(counts: []u16, prompt_tokens: []const u32, generated_tokens: []const u32, options: Options) void {
    @memset(counts, 0);
    if (options.repeat_penalty <= 1.0 or options.repeat_last_n == 0) return;

    const total = prompt_tokens.len + generated_tokens.len;
    const start = if (total > options.repeat_last_n) total - options.repeat_last_n else 0;
    var index = start;
    while (index < total) : (index += 1) {
        const token = if (index < prompt_tokens.len)
            prompt_tokens[index]
        else
            generated_tokens[index - prompt_tokens.len];
        if (token >= counts.len) continue;
        const token_index: usize = @intCast(token);
        if (counts[token_index] != std.math.maxInt(u16)) counts[token_index] += 1;
    }
}

fn greedyWithPenalties(logits: []const f32, repeat_counts: []const u16, options: Options) u32 {
    std.debug.assert(logits.len != 0);
    var best_id: u32 = 0;
    var best_logit = adjustedLogit(logits[0], repeat_counts[0], options);
    for (logits[1..], 1..) |logit, index| {
        const score = adjustedLogit(logit, repeat_counts[index], options);
        if (score > best_logit) {
            best_logit = score;
            best_id = @intCast(index);
        }
    }
    return best_id;
}

fn buildCandidates(candidates: []Candidate, logits: []const f32, repeat_counts: []const u16, options: Options) usize {
    const limit = if (options.top_k == 0) logits.len else @min(options.top_k, logits.len);
    var count: usize = 0;

    if (limit == logits.len) {
        for (logits, 0..) |logit, index| {
            const score = adjustedLogit(logit, repeat_counts[index], options);
            if (!std.math.isFinite(score)) continue;
            candidates[count] = .{ .id = @intCast(index), .logit = score };
            count += 1;
        }
        std.sort.pdq(Candidate, candidates[0..count], {}, candidateGreaterThan);
        return count;
    }

    for (logits, 0..) |logit, index| {
        const score = adjustedLogit(logit, repeat_counts[index], options);
        if (!std.math.isFinite(score)) continue;
        insertTopCandidate(candidates[0..limit], &count, .{ .id = @intCast(index), .logit = score });
    }
    return count;
}

fn adjustedLogit(logit: f32, repeat_count: u16, options: Options) f32 {
    if (repeat_count == 0 or options.repeat_penalty <= 1.0) return logit;
    return if (logit < 0.0) logit * options.repeat_penalty else logit / options.repeat_penalty;
}

fn insertTopCandidate(candidates: []Candidate, count: *usize, candidate: Candidate) void {
    if (candidates.len == 0) return;
    if (count.* < candidates.len) {
        candidates[count.*] = candidate;
        count.* += 1;
        bubbleCandidateUp(candidates[0..count.*], count.* - 1);
        return;
    }
    if (candidate.logit <= candidates[count.* - 1].logit) return;
    candidates[count.* - 1] = candidate;
    bubbleCandidateUp(candidates[0..count.*], count.* - 1);
}

fn bubbleCandidateUp(candidates: []Candidate, index: usize) void {
    var i = index;
    while (i > 0 and candidates[i].logit > candidates[i - 1].logit) : (i -= 1) {
        std.mem.swap(Candidate, &candidates[i], &candidates[i - 1]);
    }
}

fn candidateGreaterThan(_: void, lhs: Candidate, rhs: Candidate) bool {
    return lhs.logit > rhs.logit;
}

fn prepareProbabilities(candidates: []Candidate, options: Options) usize {
    if (candidates.len == 0) return 0;
    const temperature = @max(options.temperature, 1e-6);
    const max_logit = candidates[0].logit;

    var total: f32 = 0.0;
    for (candidates) |*candidate| {
        const probability = std.math.exp((candidate.logit - max_logit) / temperature);
        candidate.probability = if (std.math.isFinite(probability)) probability else 0.0;
        total += candidate.probability;
    }
    if (total <= 0.0 or !std.math.isFinite(total)) return 1;

    if (options.top_p >= 1.0) return candidates.len;

    var cumulative: f32 = 0.0;
    for (candidates, 0..) |*candidate, index| {
        cumulative += candidate.probability;
        if (cumulative / total >= options.top_p) return index + 1;
    }
    return candidates.len;
}

fn sampleCandidate(candidates: []const Candidate, random_unit: f32) u32 {
    var total: f32 = 0.0;
    for (candidates) |candidate| total += candidate.probability;
    if (total <= 0.0 or !std.math.isFinite(total)) return candidates[0].id;

    const target = @min(random_unit, 0.99999994) * total;
    var cumulative: f32 = 0.0;
    for (candidates) |candidate| {
        cumulative += candidate.probability;
        if (target <= cumulative) return candidate.id;
    }
    return candidates[candidates.len - 1].id;
}

test "greedy generation rejects empty prompts" {
    var runner: llm.Runner = undefined;
    try std.testing.expectError(Error.EmptyPrompt, generateGreedy(std.testing.allocator, &runner, &.{}, .{}));
}

test "generation stats compute throughput" {
    const stats: Stats = .{
        .prompt_tokens = 3,
        .generated_tokens = 4,
        .prefill_nanoseconds = std.time.ns_per_s,
        .decode_nanoseconds = 2 * std.time.ns_per_s,
        .total_nanoseconds = 3 * std.time.ns_per_s,
    };

    try std.testing.expectEqual(@as(f64, 2.0), stats.decodeTokensPerSecond());
}

test "embedding range selects replacement rows" {
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const range = EmbeddingRange{ .start = 2, .token_count = 2, .dimensions = 3, .data = data[0..] };
    try std.testing.expect(range.token(1) == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, range.token(2).?);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6 }, range.token(3).?);
    try std.testing.expect(range.token(4) == null);
}

test "repeat penalty can move greedy choice away from recent tokens" {
    const logits = [_]f32{ 10.0, 9.8, 1.0 };
    const counts = [_]u16{ 1, 0, 0 };
    const token = greedyWithPenalties(&logits, &counts, .{
        .temperature = 0.0,
        .repeat_penalty = 2.0,
    });
    try std.testing.expectEqual(@as(u32, 1), token);
}

test "top-k candidate builder keeps highest adjusted logits" {
    const logits = [_]f32{ 1.0, 4.0, 2.0, 3.0 };
    const counts = [_]u16{ 0, 0, 0, 0 };
    var candidates: [2]Candidate = undefined;
    const count = buildCandidates(&candidates, &logits, &counts, .{
        .top_k = 2,
        .top_p = 1.0,
    });
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u32, 1), candidates[0].id);
    try std.testing.expectEqual(@as(u32, 3), candidates[1].id);
}

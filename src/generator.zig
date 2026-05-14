// SPDX-License-Identifier: MPL-2.0

const std = @import("std");
const llm = @import("llm");

pub const Allocator = std.mem.Allocator;

pub const Error = llm.Error || error{
    EmptyPrompt,
    OutputTooSmall,
};

pub const Options = struct {
    max_new_tokens: usize = 32,
    eos_token_id: ?u32 = null,
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
    if (prompt_tokens.len == 0) return Error.EmptyPrompt;

    const out = try allocator.alloc(u32, options.max_new_tokens);
    errdefer allocator.free(out);

    const total_start = std.Io.Clock.awake.now(io);

    var position: usize = 0;
    var logits: []const f32 = &.{};
    for (prompt_tokens) |token| {
        logits = try runner.forwardToken(token, position);
        position += 1;
    }

    const prefill_done = std.Io.Clock.awake.now(io);

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

    const decode_done = std.Io.Clock.awake.now(io);
    const tokens = try allocator.realloc(out, produced);
    return .{
        .tokens = tokens,
        .stats = .{
            .prompt_tokens = prompt_tokens.len,
            .generated_tokens = produced,
            .prefill_nanoseconds = total_start.durationTo(prefill_done).toNanoseconds(),
            .decode_nanoseconds = prefill_done.durationTo(decode_done).toNanoseconds(),
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

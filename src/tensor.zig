// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const max_rank = 8;
pub const data_alignment_bytes = 64;
pub const data_alignment: std.mem.Alignment = .@"64";
pub const simd_lanes: usize = std.simd.suggestVectorLength(f32) orelse 4;

const Vec = @Vector(simd_lanes, f32);
const AlignedData = []align(data_alignment_bytes) f32;

pub const Error = error{
    InvalidShape,
    ShapeMismatch,
    LengthMismatch,
    OutOfBounds,
};

pub const Shape = struct {
    rank_value: usize = 0,
    dims_buf: [max_rank]usize = [_]usize{0} ** max_rank,

    pub fn init(shape_dims: []const usize) Error!Shape {
        if (shape_dims.len == 0 or shape_dims.len > max_rank) return Error.InvalidShape;

        var shape: Shape = .{};
        shape.rank_value = shape_dims.len;
        for (shape_dims, 0..) |dim_size, i| {
            if (dim_size == 0) return Error.InvalidShape;
            shape.dims_buf[i] = dim_size;
        }
        return shape;
    }

    pub fn rank(self: *const Shape) usize {
        return self.rank_value;
    }

    pub fn asSlice(self: *const Shape) []const usize {
        return self.dims_buf[0..self.rank_value];
    }

    pub fn dim(self: *const Shape, index: usize) Error!usize {
        if (index >= self.rank_value) return Error.OutOfBounds;
        return self.dims_buf[index];
    }

    pub fn last(self: *const Shape) Error!usize {
        if (self.rank_value == 0) return Error.InvalidShape;
        return self.dims_buf[self.rank_value - 1];
    }

    pub fn elementCount(self: *const Shape) Error!usize {
        return checkedProduct(self.asSlice());
    }

    pub fn eql(self: *const Shape, other: *const Shape) bool {
        return std.mem.eql(usize, self.asSlice(), other.asSlice());
    }
};

pub const Tensor = struct {
    data: AlignedData,
    shape: Shape,

    pub fn init(allocator: Allocator, shape_dims: []const usize) !Tensor {
        var tensor = try initUndefined(allocator, shape_dims);
        tensor.fill(0.0);
        return tensor;
    }

    pub fn initFilled(allocator: Allocator, shape_dims: []const usize, value: f32) !Tensor {
        var tensor = try initUndefined(allocator, shape_dims);
        tensor.fill(value);
        return tensor;
    }

    pub fn initUndefined(allocator: Allocator, shape_dims: []const usize) !Tensor {
        const shape = try Shape.init(shape_dims);
        const total = try shape.elementCount();
        const data = try allocator.alignedAlloc(f32, data_alignment, total);
        return .{ .data = data, .shape = shape };
    }

    pub fn fromSlice(allocator: Allocator, shape_dims: []const usize, values: []const f32) !Tensor {
        var tensor = try initUndefined(allocator, shape_dims);
        errdefer tensor.deinit(allocator);
        if (tensor.data.len != values.len) return Error.LengthMismatch;
        @memcpy(tensor.data, values);
        return tensor;
    }

    pub fn deinit(self: *Tensor, allocator: Allocator) void {
        if (self.data.len != 0) allocator.free(self.data);
        self.* = .{ .data = emptyData(), .shape = .{} };
    }

    pub fn clone(self: *const Tensor, allocator: Allocator) !Tensor {
        return fromSlice(allocator, self.dims(), self.data);
    }

    pub fn dims(self: *const Tensor) []const usize {
        return self.shape.asSlice();
    }

    pub fn rank(self: *const Tensor) usize {
        return self.shape.rank();
    }

    pub fn len(self: *const Tensor) usize {
        return self.data.len;
    }

    pub fn byteLen(self: *const Tensor) usize {
        return self.data.len * @sizeOf(f32);
    }

    pub fn asSlice(self: *Tensor) []f32 {
        return self.data;
    }

    pub fn constSlice(self: *const Tensor) []const f32 {
        return self.data;
    }

    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }

    pub fn reshape(self: *Tensor, shape_dims: []const usize) !void {
        const next_shape = try Shape.init(shape_dims);
        if (try next_shape.elementCount() != self.data.len) return Error.LengthMismatch;
        self.shape = next_shape;
    }

    pub fn row(self: *Tensor, index: usize) ![]f32 {
        try expectRank(self, 2);
        const rows = self.shape.dims_buf[0];
        const cols = self.shape.dims_buf[1];
        if (index >= rows) return Error.OutOfBounds;
        const start = index * cols;
        return self.data[start..][0..cols];
    }

    pub fn constRow(self: *const Tensor, index: usize) ![]const f32 {
        try expectRank(self, 2);
        const rows = self.shape.dims_buf[0];
        const cols = self.shape.dims_buf[1];
        if (index >= rows) return Error.OutOfBounds;
        const start = index * cols;
        return self.data[start..][0..cols];
    }

    pub fn copyFrom(self: *Tensor, source: *const Tensor) !void {
        try expectSameShape(self, source);
        @memcpy(self.data, source.data);
    }

    pub fn copyFromSlice(self: *Tensor, values: []const f32) !void {
        if (self.data.len != values.len) return Error.LengthMismatch;
        @memcpy(self.data, values);
    }

    pub fn add(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        var out = try initUndefined(allocator, a.dims());
        errdefer out.deinit(allocator);
        try addInto(&out, a, b);
        return out;
    }

    pub fn addInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        try expectSameShape(a, b);
        try expectSameShape(out, a);

        var i: usize = 0;
        while (i + simd_lanes <= out.data.len) : (i += simd_lanes) {
            const av = loadVec(a.data, i);
            const bv = loadVec(b.data, i);
            storeVec(out.data, i, av + bv);
        }
        while (i < out.data.len) : (i += 1) out.data[i] = a.data[i] + b.data[i];
    }

    pub fn sub(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        var out = try initUndefined(allocator, a.dims());
        errdefer out.deinit(allocator);
        try subInto(&out, a, b);
        return out;
    }

    pub fn subInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        try expectSameShape(a, b);
        try expectSameShape(out, a);

        var i: usize = 0;
        while (i + simd_lanes <= out.data.len) : (i += simd_lanes) {
            const av = loadVec(a.data, i);
            const bv = loadVec(b.data, i);
            storeVec(out.data, i, av - bv);
        }
        while (i < out.data.len) : (i += 1) out.data[i] = a.data[i] - b.data[i];
    }

    pub fn mul(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        var out = try initUndefined(allocator, a.dims());
        errdefer out.deinit(allocator);
        try mulInto(&out, a, b);
        return out;
    }

    pub fn mulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
        try expectSameShape(a, b);
        try expectSameShape(out, a);

        var i: usize = 0;
        while (i + simd_lanes <= out.data.len) : (i += simd_lanes) {
            const av = loadVec(a.data, i);
            const bv = loadVec(b.data, i);
            storeVec(out.data, i, av * bv);
        }
        while (i < out.data.len) : (i += 1) out.data[i] = a.data[i] * b.data[i];
    }

    pub fn mulScalarInPlace(self: *Tensor, value: f32) void {
        const vv: Vec = @splat(value);
        var i: usize = 0;
        while (i + simd_lanes <= self.data.len) : (i += simd_lanes) {
            storeVec(self.data, i, loadVec(self.data, i) * vv);
        }
        while (i < self.data.len) : (i += 1) self.data[i] *= value;
    }

    pub fn addScaledInPlace(self: *Tensor, source: *const Tensor, scale: f32) !void {
        try expectSameShape(self, source);

        const scale_v: Vec = @splat(scale);
        var i: usize = 0;
        while (i + simd_lanes <= self.data.len) : (i += simd_lanes) {
            const dst = loadVec(self.data, i);
            const src = loadVec(source.data, i);
            storeVec(self.data, i, dst + src * scale_v);
        }
        while (i < self.data.len) : (i += 1) self.data[i] = @mulAdd(f32, source.data[i], scale, self.data[i]);
    }

    pub fn reluInPlace(self: *Tensor) void {
        const zero: Vec = @splat(0.0);
        var i: usize = 0;
        while (i + simd_lanes <= self.data.len) : (i += simd_lanes) {
            const v = loadVec(self.data, i);
            storeVec(self.data, i, @max(v, zero));
        }
        while (i < self.data.len) : (i += 1) self.data[i] = @max(self.data[i], 0.0);
    }

    pub fn geluInPlace(self: *Tensor) void {
        const k = @as(f32, 0.7978845608028654);
        const c = @as(f32, 0.044715);
        for (self.data) |*value| {
            const x = value.*;
            value.* = 0.5 * x * (1.0 + std.math.tanh(k * (x + c * x * x * x)));
        }
    }

    pub fn siluInPlace(self: *Tensor) void {
        for (self.data) |*value| {
            const x = value.*;
            value.* = x / (1.0 + std.math.exp(-x));
        }
    }

    pub fn transpose2d(allocator: Allocator, input: *const Tensor) !Tensor {
        try expectRank(input, 2);
        var out = try initUndefined(allocator, &.{ input.shape.dims_buf[1], input.shape.dims_buf[0] });
        errdefer out.deinit(allocator);
        try transpose2dInto(&out, input);
        return out;
    }

    pub fn transpose2dInto(out: *Tensor, input: *const Tensor) !void {
        try expectRank(input, 2);
        try expectShape(out, &.{ input.shape.dims_buf[1], input.shape.dims_buf[0] });

        const rows = input.shape.dims_buf[0];
        const cols = input.shape.dims_buf[1];
        for (0..rows) |r| {
            for (0..cols) |c| {
                out.data[c * rows + r] = input.data[r * cols + c];
            }
        }
    }

    pub fn matmul(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        try expectRank(a, 2);
        try expectRank(b, 2);
        if (a.shape.dims_buf[1] != b.shape.dims_buf[0]) return Error.ShapeMismatch;

        const m = a.shape.dims_buf[0];
        const k = a.shape.dims_buf[1];
        const n = b.shape.dims_buf[1];

        var out = try initUndefined(allocator, &.{ m, n });
        errdefer out.deinit(allocator);

        var b_transposed = try initUndefined(allocator, &.{ n, k });
        defer b_transposed.deinit(allocator);
        try transpose2dInto(&b_transposed, b);
        try matmulInto(&out, a, b, &b_transposed);
        return out;
    }

    pub fn matmulInto(out: *Tensor, a: *const Tensor, b: *const Tensor, scratch_b_transposed: ?*Tensor) !void {
        try expectRank(a, 2);
        try expectRank(b, 2);
        if (a.shape.dims_buf[1] != b.shape.dims_buf[0]) return Error.ShapeMismatch;
        try expectShape(out, &.{ a.shape.dims_buf[0], b.shape.dims_buf[1] });

        if (scratch_b_transposed) |bt| {
            try expectShape(bt, &.{ b.shape.dims_buf[1], b.shape.dims_buf[0] });
            try transpose2dInto(bt, b);
            matmulWithTransposedB(out, a, bt);
        } else {
            matmulBlocked(out, a, b);
        }
    }

    pub fn softmaxLastDim(allocator: Allocator, input: *const Tensor) !Tensor {
        var out = try input.clone(allocator);
        errdefer out.deinit(allocator);
        try out.softmaxLastDimInPlace();
        return out;
    }

    pub fn softmaxLastDimInPlace(self: *Tensor) !void {
        const width = try self.shape.last();
        const rows = self.data.len / width;
        for (0..rows) |r| {
            const start = r * width;
            softmaxSlice(self.data[start..][0..width]);
        }
    }

    pub fn rmsNormLastDim(
        allocator: Allocator,
        input: *const Tensor,
        weight: *const Tensor,
        epsilon: f32,
    ) !Tensor {
        var out = try initUndefined(allocator, input.dims());
        errdefer out.deinit(allocator);
        try rmsNormLastDimInto(&out, input, weight, epsilon);
        return out;
    }

    pub fn rmsNormLastDimInto(
        out: *Tensor,
        input: *const Tensor,
        weight: *const Tensor,
        epsilon: f32,
    ) !void {
        try expectSameShape(out, input);
        try expectRank(weight, 1);

        const width = try input.shape.last();
        if (weight.data.len != width) return Error.ShapeMismatch;

        const rows = input.data.len / width;
        for (0..rows) |r| {
            const start = r * width;
            const in_row = input.data[start..][0..width];
            const out_row = out.data[start..][0..width];
            const mean_square = sumSquares(in_row) / @as(f32, @floatFromInt(width));
            const scale = 1.0 / @sqrt(mean_square + epsilon);
            mulWeightScaled(out_row, in_row, weight.data, scale);
        }
    }

    pub fn conv2dNchw(
        allocator: Allocator,
        input: *const Tensor,
        kernel: *const Tensor,
        bias: ?*const Tensor,
        options: Conv2DOptions,
    ) !Tensor {
        const output_dims = try conv2dOutputDims(input, kernel, bias, options);
        var out = try initUndefined(allocator, &output_dims);
        errdefer out.deinit(allocator);
        try conv2dNchwInto(&out, input, kernel, bias, options);
        return out;
    }

    pub fn conv2dNchwInto(
        out: *Tensor,
        input: *const Tensor,
        kernel: *const Tensor,
        bias: ?*const Tensor,
        options: Conv2DOptions,
    ) !void {
        const expected_dims = try conv2dOutputDims(input, kernel, bias, options);
        try expectShape(out, &expected_dims);

        const batch = input.shape.dims_buf[0];
        const in_c = input.shape.dims_buf[1];
        const in_h = input.shape.dims_buf[2];
        const in_w = input.shape.dims_buf[3];
        const out_c = kernel.shape.dims_buf[0];
        const k_h = kernel.shape.dims_buf[2];
        const k_w = kernel.shape.dims_buf[3];
        const out_h = out.shape.dims_buf[2];
        const out_w = out.shape.dims_buf[3];

        for (0..batch) |b_ix| {
            for (0..out_c) |oc| {
                for (0..out_h) |oy| {
                    for (0..out_w) |ox| {
                        var acc: f32 = if (bias) |bias_tensor| bias_tensor.data[oc] else 0.0;

                        for (0..in_c) |ic| {
                            for (0..k_h) |ky| {
                                const y = paddedIndex(oy, options.stride_h, ky, options.dilation_h, options.pad_h);
                                if (y < 0 or y >= @as(isize, @intCast(in_h))) continue;
                                const in_y: usize = @intCast(y);

                                const input_row = (((b_ix * in_c + ic) * in_h + in_y) * in_w);
                                const kernel_row = (((oc * in_c + ic) * k_h + ky) * k_w);

                                if (options.dilation_w == 1) {
                                    const x0 = paddedIndex(ox, options.stride_w, 0, 1, options.pad_w);
                                    if (x0 >= 0 and x0 + @as(isize, @intCast(k_w)) <= @as(isize, @intCast(in_w))) {
                                        const in_x: usize = @intCast(x0);
                                        acc += dotUnchecked(
                                            input.data[input_row + in_x ..][0..k_w],
                                            kernel.data[kernel_row..][0..k_w],
                                        );
                                        continue;
                                    }
                                }

                                for (0..k_w) |kx| {
                                    const x = paddedIndex(ox, options.stride_w, kx, options.dilation_w, options.pad_w);
                                    if (x < 0 or x >= @as(isize, @intCast(in_w))) continue;
                                    const in_x: usize = @intCast(x);
                                    acc = @mulAdd(
                                        f32,
                                        input.data[input_row + in_x],
                                        kernel.data[kernel_row + kx],
                                        acc,
                                    );
                                }
                            }
                        }

                        out.data[((b_ix * out_c + oc) * out_h + oy) * out_w + ox] = acc;
                    }
                }
            }
        }
    }
};

pub const Conv2DOptions = struct {
    stride_h: usize = 1,
    stride_w: usize = 1,
    pad_h: usize = 0,
    pad_w: usize = 0,
    dilation_h: usize = 1,
    dilation_w: usize = 1,
};

pub fn dot(a: []const f32, b: []const f32) Error!f32 {
    if (a.len != b.len) return Error.LengthMismatch;
    return dotUnchecked(a, b);
}

fn checkedProduct(dims: []const usize) Error!usize {
    var total: usize = 1;
    for (dims) |dim| {
        if (dim == 0) return Error.InvalidShape;
        total = std.math.mul(usize, total, dim) catch return Error.InvalidShape;
    }
    return total;
}

fn emptyData() AlignedData {
    const ptr: [*]align(data_alignment_bytes) f32 = @ptrFromInt(data_alignment_bytes);
    return ptr[0..0];
}

fn expectRank(tensor: *const Tensor, rank: usize) Error!void {
    if (tensor.rank() != rank) return Error.InvalidShape;
}

fn expectShape(tensor: *const Tensor, dims: []const usize) Error!void {
    if (!std.mem.eql(usize, tensor.dims(), dims)) return Error.ShapeMismatch;
}

fn expectSameShape(a: *const Tensor, b: *const Tensor) Error!void {
    if (!std.mem.eql(usize, a.dims(), b.dims())) return Error.ShapeMismatch;
}

inline fn loadVec(slice: []const f32, index: usize) Vec {
    return @as(Vec, slice[index..][0..simd_lanes].*);
}

inline fn storeVec(slice: []f32, index: usize, vec: Vec) void {
    inline for (0..simd_lanes) |lane| {
        slice[index + lane] = vec[lane];
    }
}

fn dotUnchecked(a: []const f32, b: []const f32) f32 {
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + simd_lanes <= a.len) : (i += simd_lanes) {
        acc += loadVec(a, i) * loadVec(b, i);
    }

    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        sum = @mulAdd(f32, a[i], b[i], sum);
    }
    return sum;
}

fn sumSquares(values: []const f32) f32 {
    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + simd_lanes <= values.len) : (i += simd_lanes) {
        const v = loadVec(values, i);
        acc += v * v;
    }

    var sum = @reduce(.Add, acc);
    while (i < values.len) : (i += 1) {
        sum = @mulAdd(f32, values[i], values[i], sum);
    }
    return sum;
}

fn mulWeightScaled(out: []f32, input: []const f32, weight: []const f32, scale: f32) void {
    const scale_v: Vec = @splat(scale);
    var i: usize = 0;
    while (i + simd_lanes <= out.len) : (i += simd_lanes) {
        storeVec(out, i, loadVec(input, i) * loadVec(weight, i) * scale_v);
    }
    while (i < out.len) : (i += 1) out[i] = input[i] * weight[i] * scale;
}

fn softmaxSlice(values: []f32) void {
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

fn matmulWithTransposedB(out: *Tensor, a: *const Tensor, b_transposed: *const Tensor) void {
    const m = a.shape.dims_buf[0];
    const k = a.shape.dims_buf[1];
    const n = b_transposed.shape.dims_buf[0];

    for (0..m) |row| {
        const a_row = a.data[row * k ..][0..k];
        for (0..n) |col| {
            const bt_row = b_transposed.data[col * k ..][0..k];
            out.data[row * n + col] = dotUnchecked(a_row, bt_row);
        }
    }
}

fn matmulBlocked(out: *Tensor, a: *const Tensor, b: *const Tensor) void {
    const tile = 32;
    const m = a.shape.dims_buf[0];
    const k = a.shape.dims_buf[1];
    const n = b.shape.dims_buf[1];

    @memset(out.data, 0.0);

    var ii: usize = 0;
    while (ii < m) : (ii += tile) {
        const i_end = @min(ii + tile, m);
        var kk: usize = 0;
        while (kk < k) : (kk += tile) {
            const k_end = @min(kk + tile, k);
            var jj: usize = 0;
            while (jj < n) : (jj += tile) {
                const j_end = @min(jj + tile, n);
                for (ii..i_end) |i| {
                    for (kk..k_end) |p| {
                        const a_ip = a.data[i * k + p];
                        for (jj..j_end) |j| {
                            out.data[i * n + j] = @mulAdd(f32, a_ip, b.data[p * n + j], out.data[i * n + j]);
                        }
                    }
                }
            }
        }
    }
}

fn conv2dOutputDims(
    input: *const Tensor,
    kernel: *const Tensor,
    bias: ?*const Tensor,
    options: Conv2DOptions,
) ![4]usize {
    try expectRank(input, 4);
    try expectRank(kernel, 4);
    if (options.stride_h == 0 or options.stride_w == 0 or options.dilation_h == 0 or options.dilation_w == 0) {
        return Error.InvalidShape;
    }

    const batch = input.shape.dims_buf[0];
    const in_c = input.shape.dims_buf[1];
    const in_h = input.shape.dims_buf[2];
    const in_w = input.shape.dims_buf[3];
    const out_c = kernel.shape.dims_buf[0];
    const kernel_in_c = kernel.shape.dims_buf[1];
    const k_h = kernel.shape.dims_buf[2];
    const k_w = kernel.shape.dims_buf[3];

    if (in_c != kernel_in_c) return Error.ShapeMismatch;
    if (bias) |bias_tensor| {
        try expectShape(bias_tensor, &.{out_c});
    }

    const out_h = try convOutputDim(in_h, k_h, options.pad_h, options.stride_h, options.dilation_h);
    const out_w = try convOutputDim(in_w, k_w, options.pad_w, options.stride_w, options.dilation_w);
    return .{ batch, out_c, out_h, out_w };
}

fn convOutputDim(input: usize, kernel: usize, pad: usize, stride: usize, dilation: usize) Error!usize {
    if (kernel == 0 or stride == 0 or dilation == 0) return Error.InvalidShape;
    const dilated_span = std.math.mul(usize, dilation, kernel - 1) catch return Error.InvalidShape;
    const effective_kernel = std.math.add(usize, dilated_span, 1) catch return Error.InvalidShape;
    const double_pad = std.math.mul(usize, pad, 2) catch return Error.InvalidShape;
    const padded_input = std.math.add(usize, input, double_pad) catch return Error.InvalidShape;
    if (padded_input < effective_kernel) return Error.InvalidShape;
    return (padded_input - effective_kernel) / stride + 1;
}

fn paddedIndex(out_index: usize, stride: usize, kernel_index: usize, dilation: usize, pad: usize) isize {
    const base = out_index * stride + kernel_index * dilation;
    return @as(isize, @intCast(base)) - @as(isize, @intCast(pad));
}

test "tensor stores shape inline and data contiguously" {
    const allocator = std.testing.allocator;
    var t = try Tensor.initFilled(allocator, &.{ 2, 3, 4 }, 2.0);
    defer t.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 24), t.len());
    try std.testing.expectEqual(@as(usize, 24 * @sizeOf(f32)), t.byteLen());
    try std.testing.expect(std.mem.eql(usize, t.dims(), &.{ 2, 3, 4 }));
    try std.testing.expect(data_alignment.check(@intFromPtr(t.data.ptr)));
    try std.testing.expectEqual(@as(f32, 2.0), t.data[23]);
}

test "matmul uses contiguous rows" {
    const allocator = std.testing.allocator;
    var a = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(allocator);

    var b = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    });
    defer b.deinit(allocator);

    var c = try Tensor.matmul(allocator, &a, &b);
    defer c.deinit(allocator);

    try std.testing.expectEqualSlices(f32, &.{ 58.0, 64.0, 139.0, 154.0 }, c.data);
}

test "softmax normalizes the last dimension" {
    const allocator = std.testing.allocator;
    var logits = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{
        1.0, 2.0, 3.0,
        1.0, 1.0, 1.0,
    });
    defer logits.deinit(allocator);

    try logits.softmaxLastDimInPlace();

    const row0 = try logits.constRow(0);
    const row1 = try logits.constRow(1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row0[0] + row0[1] + row0[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), row1[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), row1[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), row1[2], 1e-6);
}

test "rms norm works across the last dimension" {
    const allocator = std.testing.allocator;
    var input = try Tensor.fromSlice(allocator, &.{ 1, 2 }, &.{ 3.0, 4.0 });
    defer input.deinit(allocator);
    var weight = try Tensor.fromSlice(allocator, &.{2}, &.{ 1.0, 1.0 });
    defer weight.deinit(allocator);

    var out = try Tensor.rmsNormLastDim(allocator, &input, &weight, 0.0);
    defer out.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f32, 0.84852815), out.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1313708), out.data[1], 1e-6);
}

test "conv2d nchw computes a valid convolution" {
    const allocator = std.testing.allocator;
    var input = try Tensor.fromSlice(allocator, &.{ 1, 1, 3, 3 }, &.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    });
    defer input.deinit(allocator);

    var kernel = try Tensor.fromSlice(allocator, &.{ 1, 1, 2, 2 }, &.{
        1.0, 1.0,
        1.0, 1.0,
    });
    defer kernel.deinit(allocator);

    var out = try Tensor.conv2dNchw(allocator, &input, &kernel, null, .{});
    defer out.deinit(allocator);

    try std.testing.expect(std.mem.eql(usize, out.dims(), &.{ 1, 1, 2, 2 }));
    try std.testing.expectEqualSlices(f32, &.{ 12.0, 16.0, 24.0, 28.0 }, out.data);
}

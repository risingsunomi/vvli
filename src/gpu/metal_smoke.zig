// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

const Result = extern struct {
    device_name: [128]u8,
    value_count: u32,
    max_abs_error: f32,
};

extern fn vvli_metal_smoke(result: *Result) c_int;

pub fn main() !void {
    var result: Result = .{
        .device_name = [_]u8{0} ** 128,
        .value_count = 0,
        .max_abs_error = 0.0,
    };
    const rc = vvli_metal_smoke(&result);
    const name_len = std.mem.indexOfScalar(u8, &result.device_name, 0) orelse result.device_name.len;
    const name = result.device_name[0..name_len];

    if (rc != 0) {
        std.debug.print("metal smoke failed: rc={d}", .{rc});
        if (name.len != 0) std.debug.print(" device={s}", .{name});
        std.debug.print("\n", .{});
        std.process.exit(@intCast(rc));
    }

    std.debug.print(
        "metal smoke ok: device={s} values={d} max_abs_error={d:.6}\n",
        .{ name, result.value_count, result.max_abs_error },
    );
}

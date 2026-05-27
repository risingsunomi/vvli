const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dotenv_module = b.addModule("dotenv", .{
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/root.zig"),
    });

    const lib_test = b.addTest(.{ .root_module = dotenv_module });
    const run_test = b.addRunArtifact(lib_test);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_test.step);
}

// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tensor_mod = b.addModule("tensor", .{
        .root_source_file = b.path("src/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    const gguf_mod = b.addModule("gguf", .{
        .root_source_file = b.path("src/gguf.zig"),
        .target = target,
        .optimize = optimize,
    });

    const llm_mod = b.addModule("llm", .{
        .root_source_file = b.path("src/llm.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tensor", .module = tensor_mod },
            .{ .name = "gguf", .module = gguf_mod },
        },
    });

    const hf_downloader_mod = b.addModule("hf_downloader", .{
        .root_source_file = b.path("src/hf_downloader.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tokenizer_mod = b.addModule("tokenizer", .{
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "gguf", .module = gguf_mod }},
    });

    const generator_mod = b.addModule("generator", .{
        .root_source_file = b.path("src/generator.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "llm", .module = llm_mod }},
    });

    const exe = b.addExecutable(.{
        .name = "vvli",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tensor", .module = tensor_mod },
                .{ .name = "llm", .module = llm_mod },
                .{ .name = "hf_downloader", .module = hf_downloader_mod },
                .{ .name = "tokenizer", .module = tokenizer_mod },
                .{ .name = "gguf", .module = gguf_mod },
                .{ .name = "generator", .module = generator_mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const tensor_tests = b.addTest(.{
        .name = "tensor-test",
        .root_module = tensor_mod,
    });
    const run_tensor_tests = b.addRunArtifact(tensor_tests);

    const llm_tests = b.addTest(.{
        .name = "llm-test",
        .root_module = llm_mod,
    });
    const run_llm_tests = b.addRunArtifact(llm_tests);

    const hf_tests = b.addTest(.{
        .name = "hf-downloader-test",
        .root_module = hf_downloader_mod,
    });
    const run_hf_tests = b.addRunArtifact(hf_tests);

    const tokenizer_tests = b.addTest(.{
        .name = "tokenizer-test",
        .root_module = tokenizer_mod,
    });
    const run_tokenizer_tests = b.addRunArtifact(tokenizer_tests);

    const gguf_tests = b.addTest(.{
        .name = "gguf-test",
        .root_module = gguf_mod,
    });
    const run_gguf_tests = b.addRunArtifact(gguf_tests);

    const generator_tests = b.addTest(.{
        .name = "generator-test",
        .root_module = generator_mod,
    });
    const run_generator_tests = b.addRunArtifact(generator_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_tensor_tests.step);
    test_step.dependOn(&run_llm_tests.step);
    test_step.dependOn(&run_hf_tests.step);
    test_step.dependOn(&run_tokenizer_tests.step);
    test_step.dependOn(&run_gguf_tests.step);
    test_step.dependOn(&run_generator_tests.step);
}

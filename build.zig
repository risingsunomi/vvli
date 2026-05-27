// SPDX-License-Identifier: MPL-2.0

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const dotenv_mod = b.dependency("dotenv", .{
        .target = target,
        .optimize = optimize,
    }).module("dotenv");

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

    const vision_mod = b.addModule("vision", .{
        .root_source_file = b.path("src/vision.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (target.result.os.tag == .macos) {
        vision_mod.linkFramework("CoreFoundation", .{});
        vision_mod.linkFramework("CoreGraphics", .{});
        vision_mod.linkFramework("ImageIO", .{});
    }

    const vlm_mod = b.addModule("vlm", .{
        .root_source_file = b.path("src/vlm.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gguf", .module = gguf_mod },
            .{ .name = "tensor", .module = tensor_mod },
            .{ .name = "tokenizer", .module = tokenizer_mod },
            .{ .name = "vision", .module = vision_mod },
        },
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
                .{ .name = "vision", .module = vision_mod },
                .{ .name = "vlm", .module = vlm_mod },
                .{ .name = "dotenv", .module = dotenv_mod },
            },
        }),
    });

    b.installArtifact(exe);

    const smoke_metal_step = b.step("smoke-metal", "Run the Metal backend smoke test");
    if (target.result.os.tag == .macos) {
        const metal_smoke = b.addExecutable(.{
            .name = "vvli-metal-smoke",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/gpu/metal_smoke.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        metal_smoke.root_module.addCSourceFile(.{
            .file = b.path("src/gpu/metal_smoke.m"),
            .flags = &.{"-fobjc-arc"},
            .language = .objective_c,
        });
        metal_smoke.root_module.linkFramework("Foundation", .{});
        metal_smoke.root_module.linkFramework("Metal", .{});
        metal_smoke.root_module.linkFramework("CoreGraphics", .{});

        b.installArtifact(metal_smoke);
        const run_metal_smoke = b.addRunArtifact(metal_smoke);
        smoke_metal_step.dependOn(&run_metal_smoke.step);
    } else {
        smoke_metal_step.dependOn(&b.addFail("Metal smoke test is only available for macOS targets.").step);
    }

    const rocm_path = b.option([]const u8, "rocm-path", "ROCm install root") orelse "/opt/rocm";
    const rocm_clang = b.option([]const u8, "rocm-clang", "ROCm amdclang++/clang++ executable") orelse
        b.fmt("{s}/llvm/bin/amdclang++", .{rocm_path});
    const rocm_arch = b.option([]const u8, "rocm-arch", "ROCm GPU architecture, e.g. gfx1100 for Radeon RX 7900 XT") orelse "gfx1100";
    const smoke_rocm_step = b.step("smoke-rocm-llvm", "Compile the ROCm HIP smoke kernel to LLVM IR");
    const rocm_smoke = b.addSystemCommand(&.{
        rocm_clang,
        "-x",
        "hip",
        "--offload-arch",
        rocm_arch,
        "-S",
        "-emit-llvm",
        "-c",
    });
    rocm_smoke.addFileArg(b.path("src/gpu/rocm_smoke.hip"));
    rocm_smoke.addArg("-o");
    const rocm_ir = rocm_smoke.addOutputFileArg("vvli_rocm_smoke.ll");
    const install_rocm_ir = b.addInstallFile(rocm_ir, "rocm/vvli_rocm_smoke.ll");
    smoke_rocm_step.dependOn(&install_rocm_ir.step);

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

    const vision_tests = b.addTest(.{
        .name = "vision-test",
        .root_module = vision_mod,
    });
    const run_vision_tests = b.addRunArtifact(vision_tests);

    const vlm_tests = b.addTest(.{
        .name = "vlm-test",
        .root_module = vlm_mod,
    });
    const run_vlm_tests = b.addRunArtifact(vlm_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_tensor_tests.step);
    test_step.dependOn(&run_llm_tests.step);
    test_step.dependOn(&run_hf_tests.step);
    test_step.dependOn(&run_tokenizer_tests.step);
    test_step.dependOn(&run_gguf_tests.step);
    test_step.dependOn(&run_generator_tests.step);
    test_step.dependOn(&run_vision_tests.step);
    test_step.dependOn(&run_vlm_tests.step);
}

# GPU Backend Smoke Tests

VVLI still runs model inference on the CPU path. The GPU work here is backend bring-up scaffolding so kernels can be integrated behind measured smoke tests before they are used for LLM matmul.

## Metal

On macOS, the Metal smoke test builds a tiny Objective-C shim, links `Foundation`, `Metal`, and `CoreGraphics`, compiles a Metal Shading Language vector-add kernel at runtime, dispatches it on the default Metal device, and checks the result.

```sh
zig build smoke-metal
```

## ROCm / LLVM

The ROCm smoke test compiles `src/gpu/rocm_smoke.hip` through the ROCm Clang/LLVM HIP frontend and emits LLVM IR. The default target is `gfx1100`, which is the ROCm target for Radeon RX 7900 XT/XTX class RDNA3 cards.

```sh
zig build smoke-rocm-llvm -Drocm-path=/opt/rocm -Drocm-arch=gfx1100
```

If ROCm is installed somewhere else, pass `-Drocm-clang=/path/to/amdclang++`. The generated IR is installed under `zig-out/rocm/vvli_rocm_smoke.ll`.

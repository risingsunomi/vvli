# VVLI

VVLI stands for Vision, Voice, and Language Inference. It is a CPU-only Zig inference runtime for local model experiments across those domains.

The current version is focused on CPU language-model inference first: dense and MoE model plumbing, Hugging Face safetensors, GGUF metadata/tensor parsing, contiguous host memory, SIMD-friendly tensor kernels, dynamic CPU threading, KV cache reuse, and generated-token throughput reporting.

This project does not use CUDA, Metal, MPS, or any GPU backend.

## Status

Early prototype. VVLI is being generalized around model families and file formats instead of one hardcoded model. Dense Qwen2/Llama-style decoder models are the first execution target. MoE models such as OLMoE are detected as MoE and have config/sharded-weight plumbing planned around explicit expert routing instead of being forced through the dense runner.

Current development and testing is on Apple Silicon. Support and CPU tuning for other chips is planned.

## Requirements

- Zig 0.16.0 or newer
- `curl` for Hugging Face downloads
- Enough RAM for the selected model, tokenizer, scratch buffers, and KV cache

Model files downloaded from Hugging Face are governed by their upstream model licenses. This repository's license covers the vvli source code, not third-party model weights.

## Run

The default repo is currently `unsloth/Qwen2.5-0.5B-Instruct`.

```sh
zig build -Doptimize=ReleaseFast run -- --prompt "Explain CPU inference in one paragraph."
```

Explicit repo and generation settings:

```sh
zig build -Doptimize=ReleaseFast run -- \
  --repo unsloth/Qwen2.5-0.5B-Instruct \
  --prompt "Write a short note about local inference." \
  --max-new-tokens 64 \
  --ctx 512 \
  --threads 0
```

GGUF dense model example:

```sh
zig build -Doptimize=ReleaseFast run -- \
  --repo unsloth/Llama-3.2-1B-Instruct-GGUF \
  --prompt "Write a short note about local CPU inference." \
  --max-new-tokens 64
```

For `*-GGUF` repos, `auto` format mode selects a native BF16/F16/F32 `.gguf` file when one is available. You can still force a specific file:

```sh
zig build -Doptimize=ReleaseFast run -- \
  --repo unsloth/Llama-3.2-1B-Instruct-GGUF \
  --format gguf \
  --weights Llama-3.2-1B-Instruct-BF16.gguf \
  --prompt "Explain CPU inference in one paragraph."
```

The first run downloads required files into `.vvli-cache/`. Use cache-only mode after the files are present:

```sh
zig build -Doptimize=ReleaseFast run -- \
  --repo unsloth/Qwen2.5-0.5B-Instruct \
  --prompt "Hello" \
  --no-download
```

Output ends with decode throughput:

```text
[# output tokens | # tok/sec | #s decode]
```

Required downloads stream curl progress in the terminal. Weight loading also reports percent progress while tensors are copied into VVLI's contiguous CPU-owned storage.

Vision-language GGUF repos can be addressed by repo id and a local image path. The current `--image` path is native Zig-owned plumbing: VVLI validates the image, downloads/selects the text GGUF plus `mmproj`, parses both GGUF files, and reports the VLM plan before stopping at the remaining native execution kernels.

```sh
zig build -Doptimize=ReleaseFast run -- \
  --repo unsloth/Qwen3.5-9B-GGUF \
  --image ./image.jpg \
  --prompt "Describe this image in one sentence."
```

VVLI auto-selects a native BF16/F16/F32 text GGUF and `mmproj-F16.gguf` when present. Use `--weights <file>` or `--mmproj <file>` to override selection.

## CLI Options

- `--repo <owner/model>`: Hugging Face repo id
- `--revision <rev>`: Hugging Face revision, default `main`
- `--cache <dir>`: local model cache, default `.vvli-cache`
- `--format <type>`: `auto`, `safetensors`, or `gguf`
- `--weights <file>`: weight file in the repo; GGUF auto-selects BF16/F16/F32 when possible
- `--mmproj <file>`: multimodal projector file for `--image` GGUF runs
- `--prompt <text>`: prompt text
- `--image <path>`: local image path for the native vision-language path
- `--max-new-tokens <n>`: output token limit, default `64`
- `--ctx <n>`: KV cache length, default `512`
- `--threads <n>`: CPU worker count, where `0` uses host CPU count
- `--no-download`: use already cached files only
- `--raw`: do not wrap the prompt in model chat markers

## Model Targets

- Dense safetensors: Qwen2 and Llama-style decoder layouts are the current runtime path.
- GGUF: BF16/F16/F32 dense GGUF parsing and loading is the native-float boundary. Quantized GGUF files such as Q4/Q8 are rejected with `QuantizedGgufUnsupported`/`UnsupportedDType` until dequant kernels exist.
- Vision-language GGUF: repos such as `unsloth/Qwen3.5-9B-GGUF` expose image-text model metadata and separate projector files. VVLI validates `--image`, downloads/selects the text GGUF plus `mmproj`, and parses the native VLM plan. Image decode/resize, patch embedding, projector forward pass, multimodal token insertion, and Qwen3.5 text execution are still planned inside VVLI.
- MoE: OLMoE-style configs are detected via MoE fields (`num_experts`, `num_experts_per_tok`) and rejected with `MoeRuntimeUnsupported` until router/top-k expert execution is implemented.
- Sharded safetensors: repos with `model.safetensors.index.json` are rejected with `ShardedSafetensorsUnsupported` until index parsing and multi-file loading are implemented.

## Build And Test

```sh
zig build test
zig build -Doptimize=ReleaseFast test
```

## Project Layout

- `src/tensor.zig`: contiguous aligned tensor storage and CPU math kernels
- `src/llm.zig`: CPU runtime, architecture config parsing, safetensors/GGUF loading, KV cache, threaded projections
- `src/tokenizer.zig`: tokenizer loading, special-token boundaries, byte-level BPE encode/decode
- `src/gguf.zig`: GGUF metadata and tensor table parser
- `src/generator.zig`: greedy generation and throughput stats
- `src/hf_downloader.zig`: Hugging Face snapshot downloader
- `src/vision.zig`: local image path validation and future vision-language input boundary
- `src/vlm.zig`: native VLM GGUF/mmproj plan loading
- `src/main.zig`: CLI prompt runner

## License

VVLI is licensed under the Mozilla Public License 2.0 (`MPL-2.0`).

The intent is straightforward: people can use, study, modify, and distribute the project, including inside larger projects. If they distribute modifications to MPL-covered source files, those modified files must remain under MPL-2.0 and the license/copyright notices must be preserved. That keeps attribution and modification history attached to the code without forcing unrelated new files in a larger project to use the same license.

See `LICENSE` for the full terms.

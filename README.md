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

Generated text streams by default as tokens are decoded. Output ends with decode throughput:

```text
[# output tokens | # tok/sec | #s decode]
```

Generation uses CPU-side sampling by default instead of pure argmax. The default policy is `temperature=0.8`, `top_p=0.95`, `top_k=40`, and `repeat_penalty=1.10`, which avoids common greedy repetition loops on small instruction models. Use `--greedy` or `--temperature 0` for deterministic argmax debugging.

Required downloads stream curl progress in the terminal. Weight loading also reports percent progress while tensors are copied into VVLI's contiguous CPU-owned storage.

Vision-language GGUF repos can be addressed by repo id and a local image path. The current `--image` path is native Zig-owned plumbing: VVLI validates the image, downloads/selects the text GGUF plus `mmproj`, decodes/resizes the image with the native macOS ImageIO path, runs Qwen3VL patch embedding, absolute position embedding, vision transformer self-attention with M-RoPE, the Qwen-style projector MLP from `mmproj`, and builds a multimodal prompt with the right image-pad token slots.

```sh
zig build -Doptimize=ReleaseFast run -- \
  --repo unsloth/Qwen3.5-9B-GGUF \
  --weights Qwen3.5-9B-Q4_0.gguf \
  --mmproj mmproj-F16.gguf \
  --image ./image.jpg \
  --prompt "Describe this image in one sentence." \
  --ctx 8192 \
  --threads 0
```

For the current native VLM preparation test, a quantized text GGUF is enough because VVLI only reads the language GGUF metadata/tokenizer and exits successfully after printing the prepared tensor/token shapes. The generic CPU runner now has a projected-image embedding prefill hook, but full Qwen3.5 image-aware generation still needs the `qwen35` hybrid/SSM text runtime and quantized GGUF matmul support.

VVLI auto-selects a native BF16/F16/F32 text GGUF and `mmproj-F16.gguf` when present. Use `--weights <file>` or `--mmproj <file>` to override selection. The default native text selection can be very large for 9B-class VLM repos.

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
- `--temperature <f>`: sampling temperature; `0` forces greedy, default `0.8`
- `--top-p <f>`: nucleus sampling threshold, default `0.95`
- `--top-k <n>`: candidate cap before top-p; `0` disables, default `40`
- `--repeat-penalty <f>`: penalize recent token repeats; `1` disables, default `1.10`
- `--repeat-last-n <n>`: prompt/output token window for repeat penalty, default `64`
- `--seed <n>`: fixed sampling seed for reproducible output
- `--greedy`: shortcut for deterministic argmax decoding
- `--no-download`: use already cached files only
- `--stream`: stream generated text as tokens are decoded, default
- `--no-stream`: decode and print only after generation completes
- `--raw`: do not wrap the prompt in model chat markers

## Model Targets

- Dense safetensors: Qwen2 and Llama-style decoder layouts are the current runtime path.
- GGUF: BF16/F16/F32 dense GGUF parsing and loading is the native-float boundary. Quantized GGUF files such as Q4/Q8 are rejected with `QuantizedGgufUnsupported`/`UnsupportedDType` until dequant kernels exist.
- Vision-language GGUF: repos such as `unsloth/Qwen3.5-9B-GGUF` expose image-text model metadata and separate projector files. VVLI validates `--image`, downloads/selects the text GGUF plus `mmproj`, decodes/resizes images on macOS through ImageIO, runs native Qwen3VL patch embedding, position embedding, vision transformer attention with M-RoPE, runs the `mm.0`/`mm.2` projector MLP for Qwen-style merger projectors, and inserts multimodal image-pad slots into the prompt. The remaining Qwen3.5 generation boundary is the `qwen35` hybrid/SSM text runtime plus quantized GGUF matmul support.
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
- `src/vision.zig`: local image path validation plus native macOS image decode/resize
- `src/vlm.zig`: native VLM GGUF/mmproj plan loading, Qwen3VL patch embedding, vision transformer/M-RoPE, projector MLP, and image-token prompt insertion
- `src/main.zig`: CLI prompt runner
- `docs/tensor.md`: tensor API and implementation notes

## License

VVLI is licensed under the Mozilla Public License 2.0 (`MPL-2.0`).

The intent is straightforward: people can use, study, modify, and distribute the project, including inside larger projects. If they distribute modifications to MPL-covered source files, those modified files must remain under MPL-2.0 and the license/copyright notices must be preserved. That keeps attribution and modification history attached to the code without forcing unrelated new files in a larger project to use the same license.

See `LICENSE` for the full terms.

# bitrust (`bitrust-zen3`)

A lean Rust FFI inference harness and OpenAI-compatible HTTP/SSE server for [Microsoft BitNet (`bitnet.cpp`)](https://github.com/microsoft/BitNet) 1.58-bit models, tuned for AMD Zen 3 CPUs.

> **Disclaimer (`bitrust-zen3`):** This repo and its default build flags (`-march=znver3`, physical-core pinning `0,2,4,6,8,10,12,14`, and `-t 4 --threads-batch 8` split) are tuned specifically for **AMD Zen 3** (benchmarked on AMD Ryzen 7 7730U, 8C/16T, dual-channel DDR4). It can run on other x86_64 Linux CPUs by overriding `BITRUST_MARCH=native` and `--cores`, but peak speedups are verified on Zen 3.

## How It Relates to `microsoft/BitNet`

`bitrust` does **not** rewrite the 30 transformer layers in Rust. It links `bitnet.cpp`'s C/AVX2 `I2_S` kernels (`libllama.so` + `libggml.so`, compiled with `-march=znver3 -mtune=znver3 -O3`) through a thin C ABI shim (`csrc/shim.c`), and adds four practical runtime wins in Rust:

1. **Tied embedding quantization (`F16` → `Q4_K`)**: In `BitNet-b1.58-2B-4T`, the tied `token_embd.weight` (`128256 × 2560`) is `626 MB` in `F16` — over **50%** of the entire `1.20 GB` GGUF — and is scanned on every generated token for the LM head. Quantizing *only* `token_embd.weight` to `Q4_K` while keeping all 30 `I2_S` transformer layers untouched shrinks the file to **`683 MB`** and lifts the RAM-bandwidth ceiling by **+59% to +64%**.
2. **Asymmetric physical-core threading (`-t 4 --threads-batch 8`)**: Single-token decode (`tg`) saturates dual-channel DDR4 bandwidth at 4 physical Zen 3 cores (adding more cores hurts via sync overhead), while prompt eval (`pp`) scales to all 8 physical cores.
3. **`mmap` + `MADV_HUGEPAGE` pre-warm & `sched_setaffinity`**: Pre-warms weights into 2 MB transparent hugepages and pins workers strictly to physical Zen 3 cores.
4. **Lossless 4–6-gram prompt-lookup speculative decoding**: Drafts up to 4 tokens from prompt/context n-grams and verifies them in a single forward pass with exact KV-cache rollback on mismatch (bit-for-bit identical output in greedy mode).

## Benchmark Findings

Hardware: **AMD Ryzen 7 7730U** (8 physical Zen 3 cores / 16 threads, 30 GiB RAM, Fedora Linux, Clang 22.1.8, `rustc 1.99.1`).  
Model: [`microsoft/BitNet-b1.58-2B-4T`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) (`I2_S` ternary weights).

### 1. Same Unmodified `I2_S` Model (`ggml-model-i2_s.gguf`, `1.20 GB`, `F16` embeddings)

| Engine / Config | Threads | Mode | Prompt `tok/s` | Gen `tok/s` | Gain vs `bitnet.cpp` (6t) |
| --- | --- | --- | ---: | ---: | ---: |
| `bitnet.cpp` (`llama-bench`) | 6 phys | standard | `287.5` | `29.43` | baseline |
| `bitnet.cpp` (`llama-bench`) | 8 phys | standard | `322.9` | `29.24` | `-0.6%` |
| `bitnet.cpp` (`llama-bench`) | 4 phys | standard | `207.0` | `30.77` | `+4.6%` |
| `bitrust-zen3` | `-t 4 -tb 8` | standard | `311.5` | `29.75–30.40` | `+1.1–3.3%` (faster pp) |
| `bitrust-zen3` | `-t 4 -tb 8` | 4–6-gram spec (JSON/edits, 92.9% accept) | `314.1` | **`64.89`** | **`+120.5%` (2.20×)** |

### 2. Comparing `Q4_K` Tied Embeddings (`ggml-model-i2_s-embq4k.gguf`, `683 MB`, all 30 `I2_S` layers untouched)

| Engine / Config | Model Size | Mode | Prompt `tok/s` | Gen `tok/s` | Gain vs `bitnet.cpp` (6t) |
| --- | ---: | --- | ---: | ---: | ---: |
| `bitnet.cpp` (`llama-bench`, `Q8_0` embd) | `840 MB` | standard (`-t 4`) | `206.1` | `39.84` | `+35.4%` |
| `bitnet.cpp` (`llama-bench`, `Q4_K` embd) | `683 MB` | standard (`-t 4`) | `223.1` | `47.78` | `+62.4%` |
| `bitrust-zen3` (`Q4_K` embd) | `683 MB` | standard (`-t 4 -tb 8`) | `304.8` | **`48.28`** | **`+64.1%`** |
| `bitrust-zen3` (`Q4_K` embd, live HTTP) | `683 MB` | `/v1/chat/completions` | `267.7` | **`46.78`** | **`+59.0%`** |
| `bitrust-zen3` (`Q4_K` embd) | `683 MB` | 4–6-gram spec (JSON/edits, 88.9% accept) | `302.4` | **`107.35`** | **`+264.8%` (3.65×)** |

### Quality Notes (`F16` vs `Q4_K` Embeddings)

Tested side-by-side at `temperature=0.0`:
- All 30 `I2_S` transformer layers are **100% unmodified** (bit-for-bit identical).
- Coding prompts (`is_palindrome`) produced word-for-word identical output.
- Math (`17 * 24`) and factual Q&A (`why is the sky blue?`) remained accurate with minor phrasing differences from tiny logit noise on borderline tokens.
- Speculative decoding is **100% lossless** (every draft token is verified against the exact model logits with KV-cache rollback on mismatch).
- If you prefer tighter logits over maximum speed, use `--token-embedding-type q8_0` (`~40 tok/s`).

## Build & Run

### 1. Build `microsoft/BitNet` with Zen 3 flags

```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
cmake -B build-znver3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_FLAGS="-march=znver3 -mtune=znver3 -O3" \
  -DCMAKE_CXX_FLAGS="-march=znver3 -mtune=znver3 -O3"
cmake --build build-znver3 --target llama ggml llama-quantize -j "$(nproc)"
```

### 2. Quantize Only the Tied Embedding Table to `Q4_K` (Optional, +59% `tok/s`)

```bash
./build-znver3/bin/llama-quantize \
  --token-embedding-type q4_k \
  models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  models/BitNet-b1.58-2B-4T/ggml-model-i2_s-embq4k.gguf \
  I2_S 8
```

### 3. Build `bitrust`

```bash
BITNET_ROOT=/path/to/BitNet \
BITNET_BUILD_DIR=/path/to/BitNet/build-znver3 \
BITRUST_MARCH=znver3 \
cargo build --release
```

### 4. Benchmark or Serve (OpenAI-Compatible API)

```bash
# Benchmark standard vs 4-6-gram speculative lookup
./target/release/bitrust \
  --model /path/to/ggml-model-i2_s-embq4k.gguf \
  -t 4 --threads-batch 8 --cores 0,2,4,6,8,10,12,14 --draft 4 \
  bench -n 160

# Start OpenAI-compatible HTTP/SSE server on :8080
./target/release/bitrust \
  --model /path/to/ggml-model-i2_s-embq4k.gguf \
  -t 4 --threads-batch 8 --cores 0,2,4,6,8,10,12,14 --draft 4 \
  serve --host 0.0.0.0 --port 8080
```

Endpoints:
- `GET /health` — status + live `last_tok_per_sec` & `last_spec_accept_pct`
- `GET /v1/models`
- `POST /v1/chat/completions` (supports both JSON with `timings` and SSE `"stream": true`)

## Credits

- [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`) for the 1.58-bit `I2_S` kernels and `BitNet-b1.58-2B-4T` model.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for the underlying GGUF / `ggml` runtime.

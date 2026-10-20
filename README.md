# cuda-kernel-lab

Hand-written CUDA kernels for the memory-bound parts of LLM serving, **decode attention** and **MoE expert layers**, benchmarked on the same GPU against **cuDNN** fused attention, **cuBLAS** grouped GEMM, and PyTorch's **tinygemm** int4 kernel. The repo also includes smaller experiments on the Tensor Core GEMM pipeline, inline PTX `mma.sync`, softmax, INT8 GEMM, and determinism.

Every kernel is checked against an fp64 CPU reference. Negative results are kept and explained instead of deleted.

**What is not new.** None of the algorithms here are new:
- Split-KV decode with log-sum-exp merging is Flash-Decoding (Dao et al., 2023), as found in FlashInfer and vLLM's PagedAttention V2. The paged-KV page-table layout follows FlashInfer's decode API.
- Shared-prefix ("cascade") attention exists in FlashInfer, Hydragen, and ChunkAttention.
- Grouping tokens by expert and launching a grouped GEMM over an upper bound of tiles is how vLLM's `fused_moe` works. CUTLASS grouped GEMM also schedules on the device.
- Weight-only INT4/INT8 GEMM with group scales is what Marlin, tinygemm, and TensorRT-LLM do, and the fp16 "magic number" dequantization trick is standard in those kernels.
- XOR-swizzled shared memory is standard in CUTLASS/CuTe and Hopper TMA.

The point of this repo is to implement these from scratch, measure them against the strongest baselines available on the machine and against the hardware limits, and explain what bounded each version.

## Hardware and measurement method

- **GPU:** RTX 5060 Laptop GPU (Blackwell, sm_120, 26 SMs, 32 MB L2). Theoretical DRAM bandwidth is 384 GB/s; the **measured streaming-read ceiling is 336 GB/s**. CUDA 13.3, Windows 11.
- **Cold L2 for decode.** Decode kernels stream tens of MB of weights or KV, which would stay resident in the 32 MB L2 across back-to-back runs. Every decode benchmark therefore overwrites L2 before each timed run. An early version of this repo skipped this and reported a decode kernel at 671 GB/s, about 2x the physical limit.
- **Interleaved timing.** A laptop GPU's clock drifts as it heats up, so methods timed one after another are not compared fairly. All methods in one table are timed interleaved (A B C A B C …), and the median is reported. Absolute times move by about ±10% between runs; the ratios are stable.
- **Library baselines** ([`tools/torch_baselines.py`](tools/torch_baselines.py)) use the same L2 flush and median. tinygemm is called once per expert and replayed from a CUDA Graph, so Python and launch overhead are not charged to it.

Raw output of every program is in [`results/`](results/).

## Results at a glance

| # | Experiment | vs strongest baseline here | vs hardware |
|---|---|---|---|
| 01 | Paged split-KV decode attention, GQA | **0.90–1.06x cuDNN** (cuDNN gets dense KV; ours walks a scattered page table) | **88–101% of the measured DRAM ceiling** |
| 01 | + shared-prefix (cascade) path | **2.4–5.7x faster than cuDNN**, 2.2–4.7x faster than our own bandwidth-bound kernel | equivalent to reading 840–1660 GB/s without sharing (2.5–4.9x the DRAM ceiling) |
| 02 | MoE layer, FP16 grouped GEMM, no host sync | **0.85–1.14x cuBLAS grouped GEMM**, 2.0–4.5x faster than padded batched GEMM | — |
| 03 | MoE decode, weight-only INT4/INT8 | INT4: **2.4–3.5x faster than cuBLAS FP16 grouped GEMM** at ≤16 tokens. vs tinygemm int4: **1.01–1.34x at ≤16 tokens**, 0.85x at 64 tokens | INT8 at 301–338 GB/s on large experts |
| 03 | Whole MoE decode layer, end to end | **1.2–1.7x faster than a host-synced cuBLAS layer at the same FP16 precision**, 2.7–3.6x with INT4 (4-layer step, CUDA Graph) | — |
| 04 | Tensor Core GEMM: buffering × load path | best variant reaches 79% of cuBLAS | — |
| 05 | Inline PTX `mma.sync.m16n8k16` | 0/128 mismatches; WMMA wastes 50% of MACs on the same 16×8 problem | — |
| 06 | Softmax: safe / online / vec4 | — | vec4 at 82% of theoretical bandwidth |
| 07 | W8A16 INT8 GEMM with zero points | four-term expansion bit-exact; 46.1 dB SQNR | — |
| 08 | Determinism: atomics vs ordered reductions | sorted scatter_add is bit-identical and 1.66x faster than atomics (sort excluded) | — |
| 09 | A race that tests cannot find | 200 clean runs; `racecheck` finds 13,656 hazards and 1,344/2,048 wrong rows | — |
| 10 | Transpose: padding vs XOR swizzle | XOR matches padding (225 vs 224 GB/s) with no extra shared memory | — |

---

## 01 · Paged split-KV decode attention, plus a shared-prefix path

`out[b, h] = softmax(q[b, h] · K[b]ᵀ / √d) · V[b]` for one new token per sequence, with an fp16 KV cache stored in **pages** `[num_pages, 16, H_kv, 128]`. The page table uses FlashInfer's CSR convention (`kv_indptr`, `kv_indices`, `kv_last_page_len`), and the tests place pages in random order. GQA groups of G = 1, 2, 4, 8 are supported; each block serves all G query heads sharing one KV head, so every KV byte is read from DRAM once and used G times.

**Split-KV.** One block per (sequence, kv head) gives only 8 blocks at batch 1 on a 26-SM GPU. The sequence is therefore cut into chunks, each producing a partial `(o_i, lse_i)`, and a second kernel merges them with log-sum-exp weights. The chunk count comes from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and fills at most one wave. Rounding up to more blocks was measured to be worse: 1.23 waves cost 8 percentage points of bandwidth.

H_q = 32, H_kv = 8, d = 128, fp16. Time in ms, with % of theoretical DRAM bandwidth in parentheses:

| B × L | no split | v1 | v2 | v3 | cuDNN (dense KV) |
|---|---|---|---|---|---|
| 1 × 32K | 2.766 (13%) | 0.540 (65%) | 0.433 (81%) | **0.413 (85%)** | 0.440 |
| 8 × 8K | 0.903 (77%) | 1.032 (68%) | **0.905 (77%)** | 0.915 (76%) | 0.814 |
| 64 × 1K | 0.825 (85%) | 1.053 (67%) | 0.825 (85%) | **0.793 (89%)** | 0.776 |

- **v1 → v2: the bottleneck was compute latency, not DRAM.** v1 computed each (head, key) score with a warp and a 5-step shuffle reduction, a serial dependency chain that took about 8 µs per 16 KB tile against roughly 1 µs of DRAM latency. In v2 each thread computes one whole 128-wide dot product. That makes 32 threads read 32 K rows that are 256 B apart, which all fall in one bank, so the K tile is **XOR-swizzled at 16-byte granularity**: chunk `c` of row `j` is stored at `c ^ (j % 16)`. This is conflict-free and needs no padding.
- **v3 (`cp.async` double buffering) is worth at most a few percent.** v2 is already near the streaming ceiling. Earlier, on the slower v1 compute path, double buffering *lost* performance, because doubling shared memory cut occupancy from 5 to 2 blocks per SM.
- **8 × 8K stays at 77%.** Its 64 (sequence, head) pairs cannot be split evenly over 104 resident slots. Uneven, stream-K-style chunking would be the fix.

### Shared-prefix (cascade) path

With prefix caching, requests that share a system prompt or document point at the **same physical pages**. The kernel above still reads those pages once per request. It is already at the bandwidth ceiling, so the only way to go faster is to read fewer bytes:

1. Find the longest run of physical pages that every sequence starts with. This happens on the host, from the page table the caller already has; no extra API is needed.
2. **Prefix:** stack the queries of all B sequences that share a KV head (B·G rows instead of G) and attend to the shared pages once on **Tensor Cores** (WMMA). The kernel makes two passes: pass 1 computes the row max of QKᵀ, and pass 2 recomputes QKᵀ, forms P = exp(S − max), and accumulates PV in Tensor Core accumulators. Recomputing costs 50% more MMA work but avoids rescaling the output, which WMMA cannot do in registers because its accumulator layout is opaque.
3. **Suffix:** run the split-KV kernel starting at the end of the prefix.
4. **Merge** the prefix and suffix partials with the same log-sum-exp merge.

| batch (H_q = 32, H_kv = 8) | cuDNN, dense KV | split-KV v2 | cascade | vs cuDNN | vs v2 |
|---|---|---|---|---|---|
| 64 requests, shared 4K prompt, 256-token suffixes | 3.570 ms | 2.165 ms | **0.926 ms** | **3.9x** | 2.3x |
| 32 requests, shared 16K document, 512-token suffixes | 7.550 ms | 6.289 ms | **1.334 ms** | **5.7x** | 4.7x |
| 8 requests, shared 16K document, 512-token suffixes | 1.588 ms | 1.456 ms | **0.661 ms** | **2.4x** | 2.2x |

v2 already beats cuDNN on the 4K-prompt batch because the 17 MB shared prefix fits in the 32 MB L2, so the hardware dedups part of it for free. The cascade's margin grows once the shared prefix outgrows L2.

Not done: prefixes shared by only a subset of the batch (a prefix tree), and overlapping the prefix and suffix kernels on two streams.

Correctness: lengths 1, 16, 17, 32, 48, 4096, 4097, …; 3 kernel variants × {no split, heuristic, 7 chunks}; and cascade batches with partial tiles, 160 stacked queries, and G = 1/4/8. Output and LSE are both checked against fp64. A batch without shared pages correctly falls back to split-KV.

## 02 · MoE layer with a device-scheduled grouped GEMM

```
router_topk → moe_count → moe_scan → moe_permute → moe_grouped_gemm → moe_combine
```

Each expert receives a data-dependent number of rows. Instead of copying those counts to the host, the GEMM launches an **upper bound** of M-tiles (`⌈T·k/64⌉ + E`). Each block binary-searches the expert tile offsets to find its expert and exits if it has no work. With no device-to-host copy, the whole layer can be captured in a CUDA Graph. The epilogue writes only valid rows, because rows past an expert's end belong to the next expert.

| shape (fp16 in, fp32 accumulate) | cuBLAS per expert | cuBLAS padded batched | cuBLAS grouped | this repo |
|---|---|---|---|---|
| Mixtral-like prefill: T=512, E=8, k=2 | 6.74 ms | 18.95 ms (4.0x MACs) | 8.31 ms | **7.26 ms** |
| fine-grained prefill: T=512, E=64, k=6 | 1.75 ms (61 launches) | 7.42 ms (9.9x MACs) | 1.40 ms | 1.64 ms |
| fine-grained decode: T=16, E=64, k=6 | 0.658 ms | 1.090 ms | 0.526 ms | 0.543 ms |

- **Atomics are fine for allocating slots, not for accumulating floats.** `moe_permute` hands out row slots with `atomicAdd`, so row order inside an expert changes from run to run, yet the output is bit-identical. An `atomicAdd` combine differs in 47% of outputs at k = 6, but in **0%** at k = 2: 0 + a + b and 0 + b + a are bitwise equal, because the first add onto 0 is exact and float addition commutes.
- **Negative result:** a 128×128 tile with 2×2 fragments per warp was slower on every shape, with both 8- and 16-warp layouts. The next step would be `ldmatrix` + `mma.sync` with deeper pipelining, checked in Nsight Compute; GPU performance counters were not available on this machine.

## 03 · Weight-only INT4/INT8 MoE GEMM for decode

In decode, an MoE layer sees a few tokens per expert, so the expert GEMM is bounded by streaming expert weights. At fp16, cuBLAS grouped GEMM and the kernel in 02 both already run near the bandwidth ceiling, so the only way to go faster is to read fewer bytes. cuBLAS grouped GEMM has no weight-only quantized path.

- **Format:** symmetric per output column, with an fp16 scale per group of 128 along K. Scales can vary along K here (unlike the integer GEMM in 07) because weights are dequantized to fp32 inside the K loop.
- **Kernel:** GEMV-shaped. One block covers an (expert, column range) pair; 4 warps split K and reduce through shared memory; rows are processed in passes of R ∈ {1, 2, 4, 8}, templated so accumulators stay in registers. The layer is one launch with no host sync.

| decode shape | cuBLAS FP16 grouped | this repo FP16 | **W8A16** | **W4A16** | tinygemm int4 † |
|---|---|---|---|---|---|
| fine-grained E=64, T=1 | 0.125 ms | 0.119 | 0.066 (1.9x) | **0.042 (3.0x)** | 0.056 |
| fine-grained E=64, T=16 | 0.480 ms | 0.502 | 0.279 (1.7x) | **0.203 (2.4x)** | 0.225 |
| fine-grained E=64, T=64 | 0.796 ms | 1.139 | 0.643 (1.2x) | 0.574 (1.4x) | **0.488** |
| Mixtral-like E=8, T=1 | 0.712 ms | 0.676 | 0.353 (2.0x) | **0.203 (3.5x)** | 0.205 |
| Mixtral-like E=8, T=16 | 2.717 ms | 3.008 | 1.387 (2.0x) | **0.854 (3.2x)** | 0.860 |

Speedups in parentheses are vs cuBLAS FP16 grouped GEMM. Output SQNR vs fp16 is 48 dB for INT8 and 23 dB for INT4; these are round-to-nearest quantizers on random weights, and model-level accuracy is out of scope.
† tinygemm uses bf16 activations, one call per expert, replayed from a CUDA Graph. That is its best case: in real decode, routing changes every step, so a captured graph with fixed per-expert shapes cannot be reused.

How INT4 went from 206 to 301 GB/s on the Mixtral-like T=1 shape:

1. **fp16 magic-number decode + deferred scaling** (the first version measured with a cold L2: 206 GB/s). A nibble OR-ed into the mantissa of fp16 1024.0 gives 1024 + n exactly, so one bitwise op and one `HSUB2` decode two weights. Scales are applied once per 128-element group instead of once per weight.
2. **Split K across blocks** to add parallelism: within noise (−2% to +8% across shapes), so parallelism was not the main limiter.
3. **8 independent 16-byte loads in flight per thread** (→ 262 GB/s). With 1–2 rows of work per weight, a thread finishes computing long before DRAM answers, so throughput is set by loads in flight, per Little's law.
4. **Issue weight loads before the activation and scale loads of each group** (→ 301 GB/s). Otherwise every K-group started with a stall while the activations arrived.

**Where it loses:** at 64 tokens, tinygemm is 1.2x faster. Its Tensor Core path scales with rows, while this CUDA-core GEMV re-streams weights per pass of 8 rows.

### End to end: what "no host sync" buys

cuBLAS grouped GEMM takes its per-group sizes from **host** arrays. A MoE layer built on it therefore has to copy the per-expert counts to the host after routing, stall the stream, rebuild the arguments, and only then launch the GEMM, every layer. Because that host logic depends on the data, the layer also cannot be captured in a CUDA Graph. This repo's layer schedules experts on the device (as vLLM's `fused_moe` does), so it never leaves the GPU and an entire multi-layer decode step can be captured as one graph.

Full layer: route → permute → expert GEMM → combine. Fine-grained MoE (E=64, k=6, d=2048, f=1408). Each of the 4 layers has its own weights; L2 is flushed before every sample, and the methods are timed interleaved. The output of both paths is checked to match.

| time per layer, 4-layer decode step | T = 1 | T = 16 |
|---|---|---|
| cuBLAS grouped GEMM FP16, host sync every layer | 0.243 ms | 0.594 ms |
| ours FP16, launched kernel by kernel | 0.154 ms (1.58x) | 0.524 ms (1.13x) |
| **ours FP16, CUDA Graph** | **0.140 ms (1.73x)** | **0.505 ms (1.18x)** |
| **ours W4A16, CUDA Graph** | **0.067 ms (3.62x)** | **0.222 ms (2.68x)** |

At equal precision the expert GEMMs themselves cost about the same (0.119 vs 0.125 ms at T = 1), so the roughly 0.1 ms gap per layer at T = 1 is the host round trip itself. It matters most at small batch, where it is a large share of a short layer, and it shrinks as T grows. Most of the gain comes from removing the sync; graph capture adds a little more once several layers are chained.

## 04 · Tensor Core GEMM: which optimization needs which

4096³, fp16 → fp32, WMMA, 64×64×32 tiles, timed interleaved:

| | LDG + STS (through registers) | `cp.async` (global → shared) |
|---|---|---|
| single buffer | 21.08 ms | 8.48 ms |
| double buffer | 20.06 ms (**+5%**) | **6.53 ms (+30%)**, 79% of cuBLAS (5.16 ms) |

Double buffering only pays off when the copy is asynchronous.

## 05 · Inline PTX `mma.sync.m16n8k16`

The A/B/C register layouts are placed by hand from the PTX ISA tables (`groupID = lane/4`, `tig = lane%4`). The result is exact against the CPU (0/128 mismatches), and the experiment shows why the native m16n8k16 shape beats WMMA's 16×16×16 for an N = 8 problem.

## 06 · Softmax

Safe (3 passes), online (2 passes, using the same `(m, d)` merge as split-KV), and float4 variants. Tests cover N = 1, N not a multiple of the block size, odd N (vec4 skipped), 70K-wide rows, and a +500 outlier.

## 07 · W8A16 INT8 GEMM

Per-channel symmetric int8 weights and per-token asymmetric uint8 activations. The zero-point expansion `Σqa·qw − zw·Σqa − za·Σqw + K·za·zw` keeps a single term in the K loop. The integer result is bit-exact against the definition, and the full pipeline scores 46.1 dB SQNR. Scales are grouped along M and N, never K, because only a scale that is constant along k can be factored out of the sum.

## 08 · Determinism

A full-array float `atomicAdd` sum gives 5 distinct bit patterns in 5 runs; a two-pass ordered reduction gives 1. For scatter_add, `atomicAdd` differs in 20,946 of 64,000 outputs between runs, while a stable sort plus segmented sum is bit-identical and, with the sort excluded, faster (0.178 vs 0.295 ms).

## 09 · A race that tests cannot find

A block reduction ending in `__syncthreads(); return s[0];` is called twice in one kernel. 200 normal runs are all correct, but `compute-sanitizer --tool racecheck` reports 13,656 hazards, and under its scheduling 1,344 of 2,048 rows are wrong. The fix is one more barrier. Log: [`results/09_racecheck.txt`](results/09_racecheck.txt).

## 10 · Transpose: padding vs XOR swizzle

Naive: 121 GB/s; `tile[32][33]` padding: 224 GB/s; XOR swizzle `tile[r][c ^ r]`: 225 GB/s. The XOR layout needs no extra column and keeps rows 128-byte aligned, which TMA and `ldmatrix`-style loads require.

---

## Build and run

Requires CUDA 12.x/13.x and an sm_80+ GPU. The baselines also need PyTorch with CUDA.

**Windows** (MSVC Build Tools + CUDA):
```powershell
.\build.ps1                          # build everything into bin\
.\build.ps1 01_decode_attention.cu   # build + run one experiment
.\build.ps1 -Run                     # build + run everything
python tools\torch_baselines.py      # cuDNN / tinygemm baselines (after bin\03_moe_wq_decode)
```

**Linux** (Makefile provided, not yet tested on Linux):
```bash
make && make run
make racecheck
python tools/torch_baselines.py
```

Each binary prints its correctness checks and timings, and exits non-zero if any check fails.

## Layout

```
common/bench.cuh           timing (median, cold-L2, interleaved), roofline report, tolerance checks
common/moe.cuh             MoE routing / permute / combine shared by 02 and 03
01_decode_attention.cu     paged split-KV decode attention (v1/v2/v3) + shared-prefix cascade
02_moe_grouped_gemm.cu     MoE layer: routing, permute, FP16 grouped GEMM, combine
03_moe_wq_decode.cu        weight-only INT8/INT4 MoE GEMM for decode
04_gemm_pipeline.cu        Tensor Core GEMM: buffering x load path
05_mma_ptx.cu              inline PTX mma.sync vs WMMA
06_softmax.cu              safe / online / vec4 softmax
07_int8_gemm.cu            W8A16 quantization + INT8 GEMM
08_determinism.cu          atomics vs ordered reductions, sorted scatter_add
09_reduction_race.cu       shared-memory race + racecheck
10_transpose_swizzle.cu    padding vs XOR swizzle
tools/torch_baselines.py   cuDNN attention and tinygemm int4 baselines, same timing method
results/                   raw output of every program on the machine above
```

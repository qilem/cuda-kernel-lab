# cuda-kernel-lab

Hand-written CUDA kernels for LLM inference: **paged split-KV decode attention** and an **MoE expert layer with a device-scheduled grouped GEMM**, plus focused experiments on the Tensor Core GEMM pipeline, quantization, and determinism.

Every kernel is checked against an fp64 CPU reference and benchmarked against cuBLAS and/or the device roofline. Negative results are kept and explained instead of deleted.

**Hardware.** RTX 5060 Laptop GPU (Blackwell, sm_120, 26 SMs, 384 GB/s theoretical DRAM; **342 GB/s measured** streaming-read ceiling), CUDA 13.3, Windows 11. The laptop GPU throttles under sustained load, so run-to-run timings vary by roughly ±10%. Full raw logs are in [`results/`](results/).

## Results at a glance

| # | Experiment | Headline result |
|---|---|---|
| 01 | Paged split-KV decode attention (GQA, FlashInfer-style page table) | **84–91% of theoretical DRAM bandwidth**, reaching the measured streaming ceiling on 2 of 3 shapes. **6.2x** faster than no-split at 32K context |
| 02 | MoE layer: route → permute → grouped GEMM → combine | One grouped-GEMM launch, **no host sync**. **On par with cuBLAS grouped GEMM** (0.87–1.19x its time across three MoE shapes), **2.2–3.6x** faster than padded batched GEMM, bit-deterministic output |
| 03 | Tensor Core GEMM: buffering × load path | Double buffering gives **+4% with synchronous loads vs +36% with `cp.async`**. Best variant reaches **80% of cuBLAS** |
| 04 | Inline PTX `mma.sync.m16n8k16` | Hand-placed fragment layout, **0/128 mismatches**. WMMA has to pad the same 16×8 problem and wastes 50% of its MACs |
| 05 | Softmax: safe / online / vec4 | Tested at N=1, odd N, and 70K-wide rows with a +500 outlier. vec4 reaches **83% of peak BW** |
| 06 | W8A8 INT8 GEMM with zero points | Four-term expansion **bit-exact** vs definition (0/4096). **46.1 dB SQNR** vs fp64 |
| 07 | Determinism: atomics vs ordered reductions | Float `atomicAdd` sum gives 5 different results in 5 runs. Sorted scatter_add is **bit-identical** and 1.6x faster (sort excluded) |
| 08 | A race that tests can't catch | 200 clean runs, yet `racecheck` finds **13,732 hazards**, and under its scheduling **1,333/2,048** rows come out wrong |
| 09 | Transpose: padding vs XOR swizzle | XOR swizzle matches padding (225 vs 217 GB/s) **with no extra shared memory and rows kept 128 B aligned** |

---

## 01 · Paged split-KV decode attention

`out[b, h] = softmax(q[b, h] · K[b]ᵀ / √d) · V[b]`, with one query token per sequence and fp16 KV stored in a **paged cache** `[num_pages, 16, H_kv, 128]`. The page table uses the same CSR convention as FlashInfer's paged decode API (`kv_indptr`, `kv_indices`, `kv_last_page_len`), and tests use **randomly shuffled page placement** and variable sequence lengths. GQA is supported (G = 1, 2, 4, 8). Each block serves all G query heads that share a KV head, so every KV byte read from DRAM is reused G times.

**Why split-KV.** Decode has one query row per head, so "one block per (sequence, kv-head)" gives only B·H_kv blocks: 8 blocks for B=1 on a 26-SM GPU. The fix is the same as split-K for skinny GEMMs. The sequence is cut into chunks, each block produces a partial `(o_i, lse_i)`, and a second kernel merges them with log-sum-exp weights. The number of chunks is chosen from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` so that the launch fills one wave.

**How the kernel got to the bandwidth ceiling** (H_q=32, H_kv=8, d=128, fp16):

| B × L | no split | v1 split-KV | v2 + thread-per-score + XOR-swizzled K | v3 v2 + `cp.async` ×2 |
|---|---|---|---|---|
| 1 × 32K | 2.437 ms (14% BW) | 0.526 ms (66%) | **0.392 ms (89%)** | 0.384 ms (91%) |
| 8 × 8K | 0.832 ms (84%) | 1.016 ms (69%) | **0.829 ms (84%)** | 0.903 ms (78%) |
| 64 × 1K | 0.770 ms (91%) | 1.024 ms (69%) | **0.768 ms (91%)** | 0.765 ms (92%) |

What each step taught:

- **v1 was latency-bound in compute, not in DRAM.** Each SM was moving only ~10 KB/µs. With 5 resident blocks per SM, that works out to roughly 8 µs per 16 KB tile, while DRAM latency is ~1 µs. The time went into the score step, which used one warp per (head, key) dot product followed by a 5-step shuffle reduction: a serial dependency chain. Prefetching cannot help when compute is the long pole.
- **v2 gives each thread one full 128-wide dot product, with no shuffles.** That creates a new problem: 32 threads read 32 K rows that are 256 B apart, so all of them hit the same bank. The K tile is therefore stored **XOR-swizzled at 16-byte granularity**: chunk `c` of row `j` lives at chunk `c ^ (j % 16)`. This is conflict-free, needs no padding, and is undone on read. Bandwidth went from 66–69% to 84–91%.
- **v3 (`cp.async` double buffering) adds nothing on top of v2.** v2 is already at the measured streaming ceiling (342 GB/s). In the sync-score version, double buffering made things worse, because doubling shared memory cut occupancy from 5 to 2 blocks/SM.
- **Never launch past one wave.** Rounding the chunk count up instead of down was measured to be worse. For B=8 it gave 128 blocks on 104 resident slots (1.23 waves) and dropped bandwidth from 82% to 74%. The 8 × 8K case is still at 84% because its 64 (sequence, head) pairs cannot be split evenly into 104 slots. Stream-K-style uneven chunking would close that gap.

Correctness: 4 batch configurations (lengths 1, 16, 17, 32, 48, 4096, 4097, …) × 3 variants × {no split, heuristic, 7 chunks}, checking both the output and the LSE against fp64.

## 02 · MoE expert layer with a device-scheduled grouped GEMM

```
router_topk → moe_count → moe_scan → moe_permute → moe_grouped_gemm → moe_combine
 (softmax,     (tokens     (row and    (rows sorted    (all experts,     (weighted
  top-k,        per         tile        by expert)      one launch,       gather,
  renorm)       expert)     offsets)                    Tensor Cores)     fixed order)
```

Each expert receives a data-dependent number of rows. Instead of reading the counts back to the host, the grouped GEMM launches an **upper bound** on M-tiles (`⌈T·k/64⌉ + E`). Each block binary-searches the expert tile offsets to find its expert and exits if it has no work. The whole layer therefore runs without a device→host copy and can be captured in a CUDA Graph. The epilogue stages each fragment through shared memory and writes only valid rows, because rows past an expert's end belong to the next expert.

Expert GEMM stage, fp16 in / fp32 accumulate (TFLOP/s counts useful FLOPs only):

| shape | (a) cuBLAS per expert | (b) cuBLAS padded batched | (c) cuBLAS grouped | **(d) this repo** |
|---|---|---|---|---|
| Mixtral-like prefill: T=512, E=8, k=2, d=4096, f=14336 | 5.07 ms | 14.14 ms (4.0x MACs) | 6.93 ms | **6.01 ms** |
| fine-grained prefill: T=512, E=64, k=6, d=2048, f=1408 | 1.63 ms (61 launches) | 5.57 ms (9.9x MACs) | 1.30 ms | **1.56 ms** |
| fine-grained decode: T=16, E=64, k=6 | 0.597 ms | 1.047 ms | 0.468 ms | **0.473 ms** (329 GB/s of weights) |

These numbers come from one run on a cool GPU. Under sustained throttling, (d) measured anywhere from 6.0 to 8.7 ms on the Mixtral shape (cuBLAS slowed down similarly), so treat differences under ~15% between (c) and (d) as noise. The gap to per-expert cuBLAS on Mixtral-sized experts (large GEMMs, where cuBLAS picks big tuned tiles) is real.

Findings:

- **Padding is the trap under skewed routing.** With a realistic skew (one expert gets 477 rows, the mean is 48), padded batched GEMM does 9.9x the useful MACs.
- **Atomics are fine for allocating slots and not fine for accumulating floats.** `moe_permute` uses `atomicAdd` to hand out row slots, so the row order inside an expert changes from run to run. The output is still bit-identical, because each GEMM output row depends only on its own input row and combine gathers in fixed slot order. An `atomicAdd`-based combine differs in 45% of outputs at k=6. It differs in **0%** at k=2, because 0 + a + b and 0 + b + a are bitwise equal: the first add onto 0 is exact and float addition is commutative. Non-associativity only shows up from k ≥ 3.
- **Negative result: a 128×128 tile with 2×2 fragments per warp was slower on every shape**, with both 8- and 16-warp layouts. Register reuse alone did not pay for the occupancy and latency it costs with WMMA. The next step would be `ldmatrix` + `mma.sync` with swizzled shared memory and deeper pipelining, verified with Nsight Compute. Hardware counters were not available on this machine (`ERR_NVGPUCTRPERM`). The config is kept and benchmarked as row (e) in the output.
- Not done yet: the second GEMM + fused SwiGLU of a full expert FFN, and FP8/INT8 expert weights.

## 03 · Tensor Core GEMM pipeline: which optimization needs which

4096³, fp16 → fp32, WMMA, 64×64×32 tiles. Two independent axes are swept: buffering and load path.

| | LDG + STS (through registers) | `cp.async` (global → shared) |
|---|---|---|
| single buffer | 16.55 ms | 6.94 ms |
| double buffer | 15.85 ms (**+4%**) | **5.11 ms (+36%)**, 26.9 TFLOP/s = 80% of cuBLAS (4.08 ms) |

Double buffering only works when the copy is asynchronous. With synchronous loads, the "prefetch" stalls on the load before the MMA loop can start.

## 04 · Inline PTX `mma.sync.m16n8k16`

Writes the A/B/C register layout by hand from the PTX ISA tables (`groupID = lane/4`, `tig = lane%4`). It verifies exactly against CPU (0/128) and shows why the native m16n8k16 instruction beats WMMA's 16×16×16 for an N=8 problem.

## 05 · Softmax

Safe (3 passes), online (2 passes, using the same `(m, d)` merge as split-KV), and float4 variants. The tests target what breaks naive kernels: N=1, N not a multiple of the block size, odd N (vec4 path skipped), 70K-wide rows, and a +500 outlier that overflows `expf` without max subtraction.

## 06 · W8A8 INT8 GEMM

Weights use per-channel symmetric int8 and activations use per-token asymmetric uint8. The zero-point expansion `Σqa·qw − zw·Σqa − za·Σqw + K·za·zw` keeps only one term inside the K loop. Validation has two levels: the integer formula must be bit-exact against the definition, and the full pipeline is scored by SQNR (46.1 dB), not by `allclose`. Scales are grouped along M and N, never along K, because only a scale that is constant along k can be factored out of the sum.

## 07 · Determinism

Full-array sum: `atomicAdd` gives 5 distinct bit patterns in 5 runs, while a two-pass partial-sum reduction gives 1. Scatter_add (the embedding-gradient / MoE-combine pattern): `atomicAdd` differs in 20,300 of 64,000 outputs between runs, while a stable sort plus segmented sum (one block per output row) is bit-identical, and with the sort excluded it is faster (0.177 vs 0.285 ms).

## 08 · A race that tests cannot find

When a block reduction ending in `__syncthreads(); return s[0];` is called twice in one kernel (LayerNorm mean, then variance), the two inlined copies share one `__shared__` array. 200 normal runs are all correct. `compute-sanitizer --tool racecheck` reports 13,732 hazards, and under its scheduling 1,333 of 2,048 rows are wrong. The fix is to read into a register, add one more barrier, then return. Log: [`results/08_racecheck.txt`](results/08_racecheck.txt).

## 09 · Transpose: padding vs XOR swizzle

| naive | `tile[32][33]` padding | XOR swizzle `tile[r][c ^ r]` |
|---|---|---|
| 122 GB/s | 217 GB/s | 225 GB/s |

The XOR layout removes bank conflicts without the extra column. It keeps rows 128-byte aligned, which TMA and `ldmatrix`-style loads need.

---

## Build and run

Requires CUDA 12.x/13.x and a GPU with sm_80 or newer (cp.async, bf16-era Tensor Cores).

**Windows** (MSVC Build Tools + CUDA):
```powershell
.\build.ps1                          # build everything into bin\
.\build.ps1 01_decode_attention.cu   # build + run one experiment
.\build.ps1 -Run                     # build + run everything
```

**Linux** (Makefile provided, not yet tested on Linux):
```bash
make            # build all
make run        # run all
make racecheck  # compute-sanitizer on 08
```

Each binary prints its own correctness checks and timings, and exits non-zero if any check fails.

## Layout

```
common/bench.cuh            timing (median of N after warmup), roofline report, tolerance checks
01_decode_attention.cu      paged split-KV decode attention, GQA, v1/v2/v3
02_moe_grouped_gemm.cu      MoE layer: routing, permute, grouped GEMM, combine
03_gemm_pipeline.cu         Tensor Core GEMM: buffering x load path
04_mma_ptx.cu               inline PTX mma.sync vs WMMA
05_softmax.cu               safe / online / vec4 softmax
06_int8_gemm.cu             W8A8 quantization + INT8 GEMM
07_determinism.cu           atomics vs ordered reductions, sorted scatter_add
08_reduction_race.cu        shared-memory race + racecheck
09_transpose_swizzle.cu     padding vs XOR swizzle
results/                    raw output of every binary on the machine above
```

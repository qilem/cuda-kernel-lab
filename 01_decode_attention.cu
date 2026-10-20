// 01_decode_attention.cu
//
// Split-KV decode attention over a paged KV cache (flash-decoding style), with GQA.
//
//   q        : [B, Hq, D]                       fp16, one new token per sequence
//   k/v cache: [num_pages, PAGE, Hkv, D]        fp16, NHD paged layout
//   kv_indptr[B+1], kv_indices[nnz], kv_last_page_len[B]   (same CSR page-table
//              convention FlashInfer uses for its paged decode API)
//   out      : [B, Hq, D] fp16,  lse: [B, Hq] fp32
//
// Why split-KV: decode has one query row per head, so "one block per (sequence, kv head)"
// gives B*Hkv blocks. For B=1, Hkv=8 that is 8 blocks on a GPU with dozens of SMs, and a
// 32K-token context means each of those blocks streams 16 MB of KV alone. Same problem as
// a tall-skinny GEMM, same fix as split-K: cut the reduction axis (here the sequence) into
// chunks, compute a partial (o_i, lse_i) per chunk, then merge:
//
//   m = max_i lse_i,   w_i = exp(lse_i - m),   o = sum_i w_i o_i / sum_i w_i,
//   lse = m + log(sum_i w_i)
//
// Decode attention is memory-bound (AI ~= G FLOP/byte for GQA group size G), so the metric
// that matters is achieved DRAM bandwidth, not FLOP/s.
//
// GQA: the G query heads that share one KV head are handled by the same block, so every
// K/V byte loaded from DRAM is reused G times.
//
// Kernel variants (template parameter V), kept so every step in the README is reproducible:
//   V=1  one warp per (head, key) score + 5-step shuffle reduction, synchronous tile loads.
//        Per-tile compute is a long serial shuffle chain; the kernel ends up latency-bound
//        in compute, not in DRAM.
//   V=2  one THREAD per (head, key) score: a full 128-wide dot product with no shuffles.
//        32 threads then read 32 different K rows (256 B apart -> all in one bank), so the
//        K tile is stored XOR-swizzled at 16-byte granularity: chunk c of row j lives at
//        chunk c ^ (j % 16). Conflict-free, no padding.
//   V=3  V=2 + cp.async double-buffered K/V tiles.

#include "common/bench.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <numeric>
#include <random>

using namespace nvcuda;

constexpr int HEAD_DIM = 128;   // Llama / Mistral / Qwen head size
constexpr int PAGE     = 16;    // tokens per KV page
constexpr int TK       = 32;    // keys per shared-memory tile (== warp size, see softmax step)
constexpr int NT       = 128;   // threads per block; thread t owns output dim t
constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;   // 16-byte chunks per K/V row (= 16)

struct DecodeParams {
    const half* q;
    const half* k_cache;
    const half* v_cache;
    const int*  kv_indptr;
    const int*  kv_indices;
    const int*  kv_last_page_len;
    float*      part_o;      // [B*Hq, num_chunks, D]  normalized partial outputs
    float*      part_lse;    // [B*Hq, num_chunks]
    int B, Hq, Hkv, num_chunks, chunk_tokens;
    int kv_start;      // tokens skipped at the start of every sequence (cascade suffix pass)
    int chunk_base;    // this launch writes partial slots [chunk_base, chunk_base + num_chunks)
    int part_stride;   // partial slots per (sequence, q head)
    float sm_scale;
};

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One block = (KV chunk, sequence b, kv head). Grid = (num_chunks, B*Hkv).
template <int G, int V>
__global__ void __launch_bounds__(NT) decode_partial(DecodeParams p) {
    constexpr bool ASYNC   = (V == 3);
    constexpr bool SWIZZLE = (V >= 2);
    constexpr int  NBUF    = ASYNC ? 2 : 1;
    __shared__ __align__(16) half  Ks[NBUF][TK][HEAD_DIM];
    __shared__ __align__(16) half  Vs[NBUF][TK][HEAD_DIM];
    __shared__ __align__(16) float qs[G][HEAD_DIM];
    __shared__ float ps[G][TK];                 // scores, then probabilities
    __shared__ float m_s[G], d_s[G], scale_s[G];

    const int chunk = blockIdx.x;
    const int b     = blockIdx.y / p.Hkv;
    const int kvh   = blockIdx.y % p.Hkv;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;

    const int page_beg = p.kv_indptr[b];
    const int npages   = p.kv_indptr[b + 1] - page_beg;
    const int seq_len  = npages > 0 ? (npages - 1) * PAGE + p.kv_last_page_len[b] : 0;
    const int kbeg = p.kv_start + chunk * p.chunk_tokens;
    const int kend = min(seq_len, kbeg + p.chunk_tokens);

    // Gather a TK x D tile of K and V through the page table, 16 B per copy. One key row
    // is D*2 = 256 contiguous bytes, so accesses stay coalesced even though consecutive
    // pages can live anywhere in the cache. Keys past kend are zero-filled.
    auto load_tile = [&](int buf, int k0) {
        for (int c = t; c < TK * CHUNKS_PER_ROW; c += NT) {
            int j = c / CHUNKS_PER_ROW, ch = c % CHUNKS_PER_ROW;
            int kcol = (SWIZZLE ? (ch ^ (j % CHUNKS_PER_ROW)) : ch) * 8;   // K: swizzled slot
            int vcol = ch * 8;                                              // V: plain
            int key = k0 + j;
            bool ok = key < kend;
            size_t off = 0;
            if (ok) {
                int page = p.kv_indices[page_beg + key / PAGE];
                off = (((size_t)page * PAGE + key % PAGE) * p.Hkv + kvh) * HEAD_DIM + ch * 8;
            }
            if constexpr (ASYNC) {
                __pipeline_memcpy_async(&Ks[buf][j][kcol], p.k_cache + off, 16, ok ? 0 : 16);
                __pipeline_memcpy_async(&Vs[buf][j][vcol], p.v_cache + off, 16, ok ? 0 : 16);
            } else {
                uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
                if (ok) {
                    kv = *reinterpret_cast<const uint4*>(p.k_cache + off);
                    vv = *reinterpret_cast<const uint4*>(p.v_cache + off);
                }
                *reinterpret_cast<uint4*>(&Ks[buf][j][kcol]) = kv;
                *reinterpret_cast<uint4*>(&Vs[buf][j][vcol]) = vv;
            }
        }
        if constexpr (ASYNC) __pipeline_commit();
    };

    if constexpr (ASYNC) { if (kbeg < kend) load_tile(0, kbeg); }   // first tile in flight early

    // q heads [kvh*G, kvh*G + G) share this kv head; fold the softmax scale into q.
    #pragma unroll
    for (int g = 0; g < G; g++)
        qs[g][t] = __half2float(p.q[((size_t)b * p.Hq + kvh * G + g) * HEAD_DIM + t]) * p.sm_scale;
    if (t < G) { m_s[t] = -INFINITY; d_s[t] = 0.f; }
    float o[G];
    #pragma unroll
    for (int g = 0; g < G; g++) o[g] = 0.f;
    __syncthreads();

    int buf = 0;
    for (int k0 = kbeg; k0 < kend; k0 += TK) {
        // (1) bring the tile in
        if constexpr (ASYNC) {
            if (k0 + TK < kend) {
                load_tile(buf ^ 1, k0 + TK);              // prefetch the next tile...
                __pipeline_wait_prior(1);                 // ...and wait only for this one
            } else {
                __pipeline_wait_prior(0);
            }
        } else {
            load_tile(0, k0);
        }
        __syncthreads();

        // (2) scores s[g][j] = q_g . k_j
        if constexpr (V == 1) {
            // one warp per (g, j) pair, 4 dims per lane, then a 5-step shuffle reduction
            for (int pr = warp; pr < G * TK; pr += NT / 32) {
                int g = pr / TK, j = pr % TK;
                float4 qv = reinterpret_cast<const float4*>(&qs[g][0])[lane];
                uint2 kr  = reinterpret_cast<const uint2*>(&Ks[buf][j][0])[lane];
                float2 k01 = __half22float2(*reinterpret_cast<half2*>(&kr.x));
                float2 k23 = __half22float2(*reinterpret_cast<half2*>(&kr.y));
                float acc = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
                acc = warp_sum(acc);
                if (lane == 0) ps[g][j] = (k0 + j < kend) ? acc : -INFINITY;
            }
        } else {
            // one thread per (g, j) pair: a warp covers keys j = 0..31 of one head g, so the
            // q reads are broadcasts and the K reads hit 32 rows at the same logical chunk;
            // the swizzle spreads them across banks.
            for (int pr = t; pr < G * TK; pr += NT) {
                int g = pr / TK, j = pr % TK;
                const float* qg = qs[g];
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                #pragma unroll
                for (int ch = 0; ch < CHUNKS_PER_ROW; ch++) {
                    uint4 raw = *reinterpret_cast<const uint4*>(&Ks[buf][j][(ch ^ (j % CHUNKS_PER_ROW)) * 8]);
                    const half2* h = reinterpret_cast<const half2*>(&raw);
                    float2 x0 = __half22float2(h[0]), x1 = __half22float2(h[1]);
                    float2 x2 = __half22float2(h[2]), x3 = __half22float2(h[3]);
                    const float* qc = qg + ch * 8;
                    a0 += qc[0] * x0.x + qc[1] * x0.y;
                    a1 += qc[2] * x1.x + qc[3] * x1.y;
                    a2 += qc[4] * x2.x + qc[5] * x2.y;
                    a3 += qc[6] * x3.x + qc[7] * x3.y;
                }
                ps[g][j] = (k0 + j < kend) ? (a0 + a1) + (a2 + a3) : -INFINITY;
            }
        }
        __syncthreads();

        // (3) online-softmax state update, one warp per head (TK == 32: lane j owns key j).
        //     The tile always has at least one valid key, so m_new is finite.
        for (int g = warp; g < G; g += NT / 32) {
            float s     = ps[g][lane];
            float m_old = m_s[g];
            float m_new = fmaxf(m_old, warp_max(s));
            float pj    = __expf(s - m_new);               // masked keys: exp(-inf) = 0
            float sum   = warp_sum(pj);
            ps[g][lane] = pj;
            if (lane == 0) {
                float sc = __expf(m_old - m_new);          // first tile: exp(-inf) = 0
                scale_s[g] = sc;
                d_s[g] = d_s[g] * sc + sum;
                m_s[g] = m_new;
            }
        }
        __syncthreads();

        // (4) o[g][t] = o[g][t] * scale + sum_j p[g][j] * V[j][t]   (two chains for ILP)
        #pragma unroll
        for (int g = 0; g < G; g++) {
            float acc0 = o[g] * scale_s[g], acc1 = 0.f;
            #pragma unroll
            for (int j = 0; j < TK; j += 2) {
                acc0 += ps[g][j]     * __half2float(Vs[buf][j][t]);
                acc1 += ps[g][j + 1] * __half2float(Vs[buf][j + 1][t]);
            }
            o[g] = acc0 + acc1;
        }
        __syncthreads();                                   // this buffer is refilled next iteration
        if constexpr (ASYNC) buf ^= 1;
    }

    // Empty chunk (kbeg >= seq_len): d = 0 -> o = 0, lse = -inf, which the merge ignores.
    #pragma unroll
    for (int g = 0; g < G; g++) {
        size_t row = ((size_t)b * p.Hq + kvh * G + g) * p.part_stride + p.chunk_base + chunk;
        float d = d_s[g];
        p.part_o[row * HEAD_DIM + t] = d > 0.f ? o[g] / d : 0.f;
        if (t == 0) p.part_lse[row] = d > 0.f ? m_s[g] + logf(d) : -INFINITY;
    }
}

// Merge per-chunk states with log-sum-exp weights. One block per (b, q head).
__global__ void merge_states(const float* part_o, const float* part_lse,
                             half* out, float* lse_out, int num_chunks) {
    const int row = blockIdx.x, t = threadIdx.x;
    const float* lse = part_lse + (size_t)row * num_chunks;
    float m = -INFINITY;
    for (int c = 0; c < num_chunks; c++) m = fmaxf(m, lse[c]);
    float wsum = 0.f, acc = 0.f;
    if (m != -INFINITY) {
        for (int c = 0; c < num_chunks; c++) {
            float w = __expf(lse[c] - m);
            wsum += w;
            acc  += w * part_o[((size_t)row * num_chunks + c) * HEAD_DIM + t];
        }
    }
    out[(size_t)row * HEAD_DIM + t] = __float2half(wsum > 0.f ? acc / wsum : 0.f);
    if (t == 0) lse_out[row] = wsum > 0.f ? m + logf(wsum) : -INFINITY;
}

// Achievable-bandwidth reference: stream a buffer once with 16-byte loads, no compute.
__global__ void stream_read(const uint4* p, size_t n, unsigned* sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;                 // keeps the loads from being removed
}

// ------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------
template <int G> void* partial_kernel(int v) {
    switch (v) {
        case 1: return (void*)decode_partial<G, 1>;
        case 2: return (void*)decode_partial<G, 2>;
        default: return (void*)decode_partial<G, 3>;
    }
}

void* pick_kernel(int G, int v) {
    switch (G) {
        case 1: return partial_kernel<1>(v);
        case 2: return partial_kernel<2>(v);
        case 4: return partial_kernel<4>(v);
        case 8: return partial_kernel<8>(v);
    }
    fprintf(stderr, "unsupported GQA group size %d\n", G);
    exit(1);
}

// Split heuristic: launch about one full wave of resident blocks, but never make a
// chunk shorter than 256 tokens (below that the merge and per-block setup dominate).
// force_chunks > 0 overrides (used to test split vs no-split explicitly).
void plan_chunks(int G, int v, int B, int Hkv, int max_len, int force_chunks,
                 int* num_chunks, int* chunk_tokens) {
    int chunks;
    if (force_chunks > 0) {
        chunks = force_chunks;
    } else {
        int dev, sms, per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pick_kernel(G, v), NT, 0));
        // Round DOWN: never launch past one full wave. Rounding up was measured to be worse
        // (B=8, L=8K: 128 blocks on 104 resident slots = 1.23 waves -> 74% vs 82% of peak BW),
        // because the partial second wave leaves most SMs idle.
        int want = std::max(1, sms * per_sm / (B * Hkv));
        int cap  = std::max(1, ceil_div(max_len, 256));
        chunks = std::min(want, cap);
    }
    int tokens = ceil_div(ceil_div(max_len, chunks), TK) * TK;   // multiple of TK
    *chunk_tokens = tokens;
    *num_chunks   = ceil_div(max_len, tokens);
}

struct Workspace { float* part_o = nullptr; float* part_lse = nullptr; size_t cap = 0; };

void decode_attention(DecodeParams p, int G, int v, half* out, float* lse, Workspace& ws,
                      cudaStream_t s = 0) {
    size_t need = (size_t)p.B * p.Hq * p.part_stride;
    if (need > ws.cap) {
        if (ws.part_o) { CUDA_CHECK(cudaFree(ws.part_o)); CUDA_CHECK(cudaFree(ws.part_lse)); }
        CUDA_CHECK(cudaMalloc(&ws.part_o, need * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.part_lse, need * sizeof(float)));
        ws.cap = need;
    }
    p.part_o = ws.part_o;
    p.part_lse = ws.part_lse;
    dim3 grid(p.num_chunks, p.B * p.Hkv);
    void* args[] = {&p};
    CUDA_CHECK(cudaLaunchKernel(pick_kernel(G, v), grid, dim3(NT), args, 0, s));
    merge_states<<<p.B * p.Hq, HEAD_DIM, 0, s>>>(ws.part_o, ws.part_lse, out, lse, p.part_stride);
}

// A batch of variable-length sequences whose pages are scattered randomly in the cache.
struct Batch {
    int B, Hq, Hkv;
    std::vector<int> lens, indptr, indices, last;
    std::vector<half> q, k, v;
    int num_pages = 0, max_len = 0;
    size_t kv_elems = 0;

    Batch(std::vector<int> lens_, int Hq_, int Hkv_, unsigned seed, bool fill = true)
        : B((int)lens_.size()), Hq(Hq_), Hkv(Hkv_), lens(lens_) {
        indptr.push_back(0);
        for (int L : lens) {
            int np = ceil_div(L, PAGE);
            indptr.push_back(indptr.back() + np);
            last.push_back(L - (np - 1) * PAGE);
            max_len = std::max(max_len, L);
        }
        num_pages = indptr.back();
        indices.resize(num_pages);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);   // non-contiguous page placement

        kv_elems = (size_t)num_pages * PAGE * Hkv * HEAD_DIM;
        q.resize((size_t)B * Hq * HEAD_DIM);
        if (fill) {
            k.resize(kv_elems);
            v.resize(kv_elems);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            for (auto& x : q) x = __float2half(U(rng));
            for (auto& x : k) x = __float2half(U(rng));
            for (auto& x : v) x = __float2half(U(rng));
        }
    }
    size_t kv_bytes() const { return kv_elems * sizeof(half); }
};

// fp64 reference straight from the definition, walking the same page table.
void cpu_reference(const Batch& bt, std::vector<float>& out, std::vector<float>& lse) {
    const int G = bt.Hq / bt.Hkv;
    const double scale = 1.0 / std::sqrt((double)HEAD_DIM);
    out.assign((size_t)bt.B * bt.Hq * HEAD_DIM, 0.f);
    lse.assign((size_t)bt.B * bt.Hq, 0.f);
    for (int b = 0; b < bt.B; b++)
        for (int h = 0; h < bt.Hq; h++) {
            int kvh = h / G, L = bt.lens[b];
            const half* qv = &bt.q[((size_t)b * bt.Hq + h) * HEAD_DIM];
            std::vector<double> s(L);
            double m = -1e300;
            for (int key = 0; key < L; key++) {
                int page = bt.indices[bt.indptr[b] + key / PAGE];
                size_t off = (((size_t)page * PAGE + key % PAGE) * bt.Hkv + kvh) * HEAD_DIM;
                double acc = 0;
                for (int d = 0; d < HEAD_DIM; d++) acc += (double)__half2float(qv[d]) * __half2float(bt.k[off + d]);
                s[key] = acc * scale;
                m = std::max(m, s[key]);
            }
            double den = 0;
            for (int key = 0; key < L; key++) { s[key] = std::exp(s[key] - m); den += s[key]; }
            for (int key = 0; key < L; key++) {
                int page = bt.indices[bt.indptr[b] + key / PAGE];
                size_t off = (((size_t)page * PAGE + key % PAGE) * bt.Hkv + kvh) * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++)
                    out[((size_t)b * bt.Hq + h) * HEAD_DIM + d] += (float)(s[key] / den * __half2float(bt.v[off + d]));
            }
            lse[(size_t)b * bt.Hq + h] = (float)(m + std::log(den));
        }
}

struct DeviceBatch {
    half *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
    int *indptr = nullptr, *indices = nullptr, *last = nullptr;
    float* lse = nullptr;
    explicit DeviceBatch(const Batch& bt) {
        CUDA_CHECK(cudaMalloc(&q, bt.q.size() * 2));
        CUDA_CHECK(cudaMalloc(&k, bt.kv_bytes()));
        CUDA_CHECK(cudaMalloc(&v, bt.kv_bytes()));
        CUDA_CHECK(cudaMalloc(&out, bt.q.size() * 2));
        CUDA_CHECK(cudaMalloc(&lse, (size_t)bt.B * bt.Hq * 4));
        CUDA_CHECK(cudaMalloc(&indptr, bt.indptr.size() * 4));
        CUDA_CHECK(cudaMalloc(&indices, std::max<size_t>(1, bt.indices.size()) * 4));
        CUDA_CHECK(cudaMalloc(&last, bt.last.size() * 4));
        CUDA_CHECK(cudaMemcpy(indptr, bt.indptr.data(), bt.indptr.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(indices, bt.indices.data(), bt.indices.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(last, bt.last.data(), bt.last.size() * 4, cudaMemcpyHostToDevice));
        if (!bt.k.empty()) {
            CUDA_CHECK(cudaMemcpy(q, bt.q.data(), bt.q.size() * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(k, bt.k.data(), bt.kv_bytes(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(v, bt.v.data(), bt.kv_bytes(), cudaMemcpyHostToDevice));
        } else {                                           // perf runs: contents don't matter
            CUDA_CHECK(cudaMemset(q, 0, bt.q.size() * 2));
            CUDA_CHECK(cudaMemset(k, 0, bt.kv_bytes()));
            CUDA_CHECK(cudaMemset(v, 0, bt.kv_bytes()));
        }
    }
    ~DeviceBatch() {
        cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(out); cudaFree(lse);
        cudaFree(indptr); cudaFree(indices); cudaFree(last);
    }
    DecodeParams params(const Batch& bt, int force_chunks, int variant) const {
        DecodeParams p{};
        p.q = q; p.k_cache = k; p.v_cache = v;
        p.kv_indptr = indptr; p.kv_indices = indices; p.kv_last_page_len = last;
        p.B = bt.B; p.Hq = bt.Hq; p.Hkv = bt.Hkv;
        p.sm_scale = 1.0f / std::sqrt((float)HEAD_DIM);
        plan_chunks(bt.Hq / bt.Hkv, variant, bt.B, bt.Hkv, bt.max_len, force_chunks, &p.num_chunks, &p.chunk_tokens);
        p.kv_start = 0;
        p.chunk_base = 0;
        p.part_stride = p.num_chunks;
        return p;
    }
};

// ====================================================================================
// Shared-prefix (cascade) decode
// ====================================================================================
// With prefix caching, requests that share a system prompt point at the SAME physical KV
// pages. The split-KV kernel above still reads those pages once per request. It already runs
// at the DRAM bandwidth ceiling, so the only way to go faster is to read fewer bytes:
//
//   1. find the longest run of physical pages that every sequence starts with (host, from the
//      page table the caller already has -- no extra API)
//   2. prefix: stack the queries of ALL sequences that share a KV head and attend to the shared
//      pages once, on Tensor Cores (B*G query rows instead of G -> the work becomes GEMM-like)
//   3. suffix: the split-KV kernel above, starting at the end of the prefix
//   4. merge prefix and suffix partials with the same log-sum-exp merge as split-KV
//
// The prefix kernel makes two passes over its key chunk: pass 1 computes the row max with
// Q K^T, pass 2 recomputes Q K^T, forms P = exp(S - max) and accumulates P V in Tensor Core
// accumulators. Recomputing Q K^T costs 50% more MMA work but means the output never has to
// be rescaled, which WMMA cannot do in registers (its accumulator layout is opaque).
//
// Prior work: FlashInfer cascade attention, Hydragen, ChunkAttention. Not new.

constexpr int CQ  = 64;               // query rows per block (4 warps x 16)
constexpr int CTK = 32;               // keys per tile
constexpr int LDK = HEAD_DIM + 8;     // 136 halves: K/V tile rows, keeps WMMA loads 32 B aligned
constexpr int LDS = CTK + 4;          // fp32 score tile
constexpr int LDP = CTK + 8;          // fp16 probability tile

struct PrefixParams {
    const half* q;
    const half* k_cache;
    const half* v_cache;
    const int*  prefix_pages;         // physical pages of the shared prefix, in order
    float*      part_o;
    float*      part_lse;
    int B, Hq, Hkv, prefix_len, chunk_tokens, part_stride;
    float sm_scale;
};

// grid = (prefix chunks, ceil(B*G / CQ), Hkv); block = 128 threads.
// Writes partial (o, lse) for chunk slot blockIdx.x of every (sequence, q head) it covers.
__global__ void __launch_bounds__(128) cascade_prefix(PrefixParams p) {
    __shared__ __align__(32) half  Qs[CQ][HEAD_DIM];
    __shared__ __align__(32) half  Ks[CTK][LDK];
    __shared__ __align__(32) half  Vs[CTK][LDK];
    __shared__ __align__(32) float Ss[CQ][LDS];
    __shared__ __align__(32) half  Ps[CQ][LDP];
    __shared__ float m_s[CQ], l_s[CQ];

    const int chunk = blockIdx.x, qtile = blockIdx.y, kvh = blockIdx.z;
    const int G = p.Hq / p.Hkv, nq = p.B * G;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int kbeg = chunk * p.chunk_tokens;
    const int kend = min(p.prefix_len, kbeg + p.chunk_tokens);

    // Q tile: row r is query qi = qtile*CQ + r = (sequence b, group member g); pre-scaled
    for (int i = t; i < CQ * CHUNKS_PER_ROW; i += 128) {
        int r = i / CHUNKS_PER_ROW, c8 = (i % CHUNKS_PER_ROW) * 8;
        int qi = qtile * CQ + r;
        __align__(16) half tmp[8];
        if (qi < nq) {
            int b = qi / G, g = qi % G;
            uint4 raw = *reinterpret_cast<const uint4*>(p.q + ((size_t)b * p.Hq + kvh * G + g) * HEAD_DIM + c8);
            const half* hv = reinterpret_cast<const half*>(&raw);
            #pragma unroll
            for (int j = 0; j < 8; j++) tmp[j] = __float2half(__half2float(hv[j]) * p.sm_scale);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) tmp[j] = __float2half(0.f);
        }
        *reinterpret_cast<uint4*>(&Qs[r][c8]) = *reinterpret_cast<uint4*>(tmp);
    }
    if (t < CQ) { m_s[t] = -INFINITY; l_s[t] = 0.f; }
    __syncthreads();

    // this warp's 16 query rows stay in registers for the whole chunk
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[HEAD_DIM / 16];
    #pragma unroll
    for (int kk = 0; kk < HEAD_DIM / 16; kk++) wmma::load_matrix_sync(qf[kk], &Qs[warp * 16][kk * 16], HEAD_DIM);

    auto load_tile = [&](int k0, bool with_v) {
        for (int c = t; c < CTK * CHUNKS_PER_ROW; c += 128) {
            int j = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * 8;
            int key = k0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (key < kend) {
                int page = p.prefix_pages[key / PAGE];
                size_t off = (((size_t)page * PAGE + key % PAGE) * p.Hkv + kvh) * HEAD_DIM + col;
                kv = *reinterpret_cast<const uint4*>(p.k_cache + off);
                if (with_v) vv = *reinterpret_cast<const uint4*>(p.v_cache + off);
            }
            *reinterpret_cast<uint4*>(&Ks[j][col]) = kv;
            if (with_v) *reinterpret_cast<uint4*>(&Vs[j][col]) = vv;
        }
    };

    // S[16 rows of this warp][CTK keys] = Q K^T. K stored [key][dim] read as col_major = K^T.
    auto scores = [&]() {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
        #pragma unroll
        for (int n = 0; n < CTK / 16; n++) {
            wmma::fill_fragment(sacc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < HEAD_DIM / 16; kk++) {
                wmma::load_matrix_sync(kf, &Ks[n * 16][kk * 16], LDK);
                wmma::mma_sync(sacc, qf[kk], kf, sacc);
            }
            wmma::store_matrix_sync(&Ss[warp * 16][n * 16], sacc, LDS, wmma::mem_row_major);
        }
        __syncwarp();
    };

    // a lane pair owns one of the warp's 16 rows, 16 keys each
    const int row = warp * 16 + (lane >> 1), c0 = (lane & 1) * 16;

    // pass 1: row max
    for (int k0 = kbeg; k0 < kend; k0 += CTK) {
        load_tile(k0, false);
        __syncthreads();
        scores();
        float mx = -INFINITY;
        #pragma unroll
        for (int c = 0; c < 16; c++)
            if (k0 + c0 + c < kend) mx = fmaxf(mx, Ss[row][c0 + c]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        if ((lane & 1) == 0) m_s[row] = fmaxf(m_s[row], mx);
        __syncthreads();
    }

    // pass 2: P = exp(S - max), l += rowsum(P), O += P V
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[HEAD_DIM / 16];
    #pragma unroll
    for (int j = 0; j < HEAD_DIM / 16; j++) wmma::fill_fragment(oacc[j], 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
    for (int k0 = kbeg; k0 < kend; k0 += CTK) {
        load_tile(k0, true);
        __syncthreads();
        scores();
        float m = m_s[row], sum = 0.f;
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            float pv = (k0 + c0 + c < kend) ? __expf(Ss[row][c0 + c] - m) : 0.f;
            Ps[row][c0 + c] = __float2half(pv);
            sum += pv;
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        if ((lane & 1) == 0) l_s[row] += sum;
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < CTK; kk += 16) {
            wmma::load_matrix_sync(pf, &Ps[warp * 16][kk], LDP);
            #pragma unroll
            for (int j = 0; j < HEAD_DIM / 16; j++) {
                wmma::load_matrix_sync(vf, &Vs[kk][j * 16], LDK);
                wmma::mma_sync(oacc[j], pf, vf, oacc[j]);
            }
        }
        __syncthreads();
    }

    // normalized partial output, staged one 16x16 fragment at a time through this warp's Ss rows
    float* stg = &Ss[warp * 16][0];
    #pragma unroll
    for (int j = 0; j < HEAD_DIM / 16; j++) {
        wmma::store_matrix_sync(stg, oacc[j], LDS, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            int rr = i / 16, cc = i % 16, r = warp * 16 + rr, qi = qtile * CQ + r;
            if (qi < nq) {
                int b = qi / G, g = qi % G;
                size_t prow = ((size_t)b * p.Hq + kvh * G + g) * p.part_stride + chunk;
                float l = l_s[r];
                p.part_o[prow * HEAD_DIM + j * 16 + cc] = l > 0.f ? stg[rr * LDS + cc] / l : 0.f;
            }
        }
        __syncwarp();
    }
    if (t < CQ) {
        int qi = qtile * CQ + t;
        if (qi < nq) {
            int b = qi / G, g = qi % G;
            size_t prow = ((size_t)b * p.Hq + kvh * G + g) * p.part_stride + chunk;
            p.part_lse[prow] = l_s[t] > 0.f ? m_s[t] + logf(l_s[t]) : -INFINITY;
        }
    }
}

// Longest run of physical pages that EVERY sequence starts with. Each sequence keeps at least
// its last page private, so every shared page is full and every suffix is non-empty.
int shared_prefix_pages(const std::vector<int>& indptr, const std::vector<int>& indices) {
    int B = (int)indptr.size() - 1;
    if (B < 2) return 0;
    for (int P = 0;; P++) {
        for (int b = 0; b < B; b++)
            if (P >= indptr[b + 1] - indptr[b] - 1) return P;
        int pg = indices[indptr[0] + P];
        for (int b = 1; b < B; b++)
            if (indices[indptr[b] + P] != pg) return P;
    }
}

void ensure_workspace(Workspace& ws, size_t need) {
    if (need <= ws.cap) return;
    if (ws.part_o) { CUDA_CHECK(cudaFree(ws.part_o)); CUDA_CHECK(cudaFree(ws.part_lse)); }
    CUDA_CHECK(cudaMalloc(&ws.part_o, need * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.part_lse, need * sizeof(float)));
    ws.cap = need;
}

struct CascadePlan {
    bool use = false;
    PrefixParams pre{};
    dim3 pre_grid;
    DecodeParams suf{};
    int G = 1;
};

// Builds the cascade launch plan from the page table, or returns use=false when the shared
// prefix is shorter than min_prefix tokens (then the plain split-KV path is used).
CascadePlan plan_cascade(const DecodeParams& base, const std::vector<int>& indptr,
                         const std::vector<int>& indices, const std::vector<int>& lens,
                         const int* d_indices, int min_prefix = 256) {
    CascadePlan cp;
    int pages = shared_prefix_pages(indptr, indices);
    int Lp = pages * PAGE;
    if (Lp < min_prefix || Lp == 0) return cp;
    cp.use = true;
    cp.G = base.Hq / base.Hkv;
    int nq = base.B * cp.G, qtiles = ceil_div(nq, CQ);

    int dev, sms, per_sm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, cascade_prefix, 128, 0));
    int cpre = std::max(1, sms * per_sm / (qtiles * base.Hkv));     // at most one wave
    cpre = std::min(cpre, std::max(1, Lp / 256));
    int ctok = ceil_div(ceil_div(Lp, cpre), CTK) * CTK;
    cpre = ceil_div(Lp, ctok);

    int max_suffix = 0;
    for (int L : lens) max_suffix = std::max(max_suffix, L - Lp);
    cp.suf = base;
    plan_chunks(cp.G, 2, base.B, base.Hkv, max_suffix, 0, &cp.suf.num_chunks, &cp.suf.chunk_tokens);
    cp.suf.kv_start = Lp;
    cp.suf.chunk_base = cpre;
    cp.suf.part_stride = cpre + cp.suf.num_chunks;

    cp.pre.q = base.q; cp.pre.k_cache = base.k_cache; cp.pre.v_cache = base.v_cache;
    cp.pre.prefix_pages = d_indices + indptr[0];
    cp.pre.B = base.B; cp.pre.Hq = base.Hq; cp.pre.Hkv = base.Hkv;
    cp.pre.prefix_len = Lp; cp.pre.chunk_tokens = ctok; cp.pre.part_stride = cp.suf.part_stride;
    cp.pre.sm_scale = base.sm_scale;
    cp.pre_grid = dim3(cpre, qtiles, base.Hkv);
    return cp;
}

void cascade_decode(CascadePlan cp, half* out, float* lse, Workspace& ws, cudaStream_t s = 0) {
    size_t need = (size_t)cp.suf.B * cp.suf.Hq * cp.suf.part_stride;
    ensure_workspace(ws, need);
    cp.pre.part_o = cp.suf.part_o = ws.part_o;
    cp.pre.part_lse = cp.suf.part_lse = ws.part_lse;
    cascade_prefix<<<cp.pre_grid, 128, 0, s>>>(cp.pre);
    void* args[] = {&cp.suf};
    CUDA_CHECK(cudaLaunchKernel(pick_kernel(cp.G, 2), dim3(cp.suf.num_chunks, cp.suf.B * cp.suf.Hkv), dim3(NT), args, 0, s));
    merge_states<<<cp.suf.B * cp.suf.Hq, HEAD_DIM, 0, s>>>(ws.part_o, ws.part_lse, out, lse, cp.suf.part_stride);
}

// B sequences that share the first Lp tokens (same physical pages, as with prefix caching),
// followed by private suffixes of the given lengths. All pages are scattered randomly.
Batch make_shared_prefix_batch(int Lp, const std::vector<int>& suffix, int Hq, int Hkv, unsigned seed, bool fill) {
    std::vector<int> lens;
    for (int s : suffix) lens.push_back(Lp + s);
    Batch bt(lens, Hq, Hkv, seed, false);
    int pp = Lp / PAGE;
    int total = pp;
    for (size_t b = 0; b < lens.size(); b++) total += (bt.indptr[b + 1] - bt.indptr[b]) - pp;
    std::vector<int> perm(total);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed + 1);
    std::shuffle(perm.begin(), perm.end(), rng);
    int next = pp;
    for (size_t b = 0; b < lens.size(); b++)
        for (int i = 0; i < bt.indptr[b + 1] - bt.indptr[b]; i++)
            bt.indices[bt.indptr[b] + i] = (i < pp) ? perm[i] : perm[next++];
    bt.num_pages = total;
    bt.kv_elems = (size_t)total * PAGE * Hkv * HEAD_DIM;
    if (fill) {
        bt.k.resize(bt.kv_elems);
        bt.v.resize(bt.kv_elems);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (auto& x : bt.q) x = __float2half(U(rng));
        for (auto& x : bt.k) x = __float2half(U(rng));
        for (auto& x : bt.v) x = __float2half(U(rng));
    }
    return bt;
}

int main() {
    Bench bench;
    Workspace ws;

    // ---------------- correctness ----------------
    struct Case { std::vector<int> lens; int Hq, Hkv; const char* tag; };
    std::vector<Case> cases = {
        {{1, 17, 1000},          8,  8, "MHA, len 1 / 17 / 1000"},
        {{5, 300, 2049, 4096},  32,  8, "GQA G=4, mixed lengths"},
        {{33, 777},             16,  2, "GQA G=8"},
        {{16, 32, 48, 4097},     4,  2, "page-boundary lengths, G=2"},
    };
    printf("\n[correctness vs fp64 reference: output (fp16) and LSE, variants v1/v2/v3 x 3 split settings]\n");
    for (auto& c : cases) {
        Batch bt(c.lens, c.Hq, c.Hkv, 7);
        std::vector<float> ref_o, ref_lse;
        cpu_reference(bt, ref_o, ref_lse);
        DeviceBatch db(bt);
        for (int v = 1; v <= 3; v++)
            for (int force : {1, 0, 7}) {                  // no split / heuristic / odd split count
                DecodeParams p = db.params(bt, force, v);
                decode_attention(p, c.Hq / c.Hkv, v, db.out, db.lse, ws);
                CUDA_CHECK_KERNEL();
                std::vector<half> ho(bt.q.size());
                std::vector<float> got_o(bt.q.size()), got_lse((size_t)bt.B * bt.Hq);
                CUDA_CHECK(cudaMemcpy(ho.data(), db.out, ho.size() * 2, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(got_lse.data(), db.lse, got_lse.size() * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < ho.size(); i++) got_o[i] = __half2float(ho[i]);
                char tag[128];
                snprintf(tag, sizeof tag, "%-27s v%d chunks=%-3d out", c.tag, v, p.num_chunks);
                bench.verify(got_o.data(), ref_o.data(), got_o.size(), tag, 1e-2f, 2e-3f);
                snprintf(tag, sizeof tag, "%-27s v%d chunks=%-3d lse", c.tag, v, p.num_chunks);
                bench.verify(got_lse.data(), ref_lse.data(), got_lse.size(), tag, 1e-4f, 1e-4f);
            }
    }

    // ---------------- correctness: shared-prefix (cascade) path ----------------
    struct SCase { int Lp; std::vector<int> suffix; int Hq, Hkv; const char* tag; };
    std::vector<SCase> scases = {
        {64,   {1, 17, 40},              8, 8, "MHA, prefix 64, suffix 1/17/40"},
        {48,   {5, 300, 33, 16, 2},     32, 8, "G=4, prefix 48 (partial tile)"},
        {1024, std::vector<int>(40, 77), 32, 8, "G=4, 40 seqs = 160 queries"},
        {256,  {100, 3},                16, 2, "G=8"},
    };
    printf("\n[correctness: cascade (shared prefix on Tensor Cores + split-KV suffix) vs fp64]\n");
    for (auto& c : scases) {
        Batch bt = make_shared_prefix_batch(c.Lp, c.suffix, c.Hq, c.Hkv, 13, true);
        std::vector<float> ref_o, ref_lse;
        cpu_reference(bt, ref_o, ref_lse);
        DeviceBatch db(bt);
        DecodeParams base = db.params(bt, 0, 2);
        CascadePlan cp = plan_cascade(base, bt.indptr, bt.indices, bt.lens, db.indices, /*min_prefix=*/16);
        if (!cp.use || cp.suf.kv_start != c.Lp) { printf("  [FAIL] %s: shared prefix not detected\n", c.tag); bench.n_fail++; continue; }
        cascade_decode(cp, db.out, db.lse, ws);
        CUDA_CHECK_KERNEL();
        std::vector<half> ho(bt.q.size());
        std::vector<float> got_o(bt.q.size()), got_lse((size_t)bt.B * bt.Hq);
        CUDA_CHECK(cudaMemcpy(ho.data(), db.out, ho.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_lse.data(), db.lse, got_lse.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < ho.size(); i++) got_o[i] = __half2float(ho[i]);
        char tag[128];
        snprintf(tag, sizeof tag, "%-32s prefix %d tok, out", c.tag, cp.suf.kv_start);
        bench.verify(got_o.data(), ref_o.data(), got_o.size(), tag, 1e-2f, 3e-3f);
        snprintf(tag, sizeof tag, "%-32s prefix %d tok, lse", c.tag, cp.suf.kv_start);
        bench.verify(got_lse.data(), ref_lse.data(), got_lse.size(), tag, 1e-3f, 2e-3f);
    }
    {   // a batch with no shared pages must fall back to plain split-KV
        Batch bt(std::vector<int>(4, 1000), 32, 8, 3, false);
        int pages = shared_prefix_pages(bt.indptr, bt.indices);
        printf("  %s  unshared batch: detected shared pages = %d\n", pages == 0 ? "[ok]  " : "[FAIL]", pages);
        if (pages) bench.n_fail++;
    }

    // ---------------- achievable bandwidth ----------------
    double ceiling_GBs;
    {
        size_t bytes = (size_t)256 << 20;
        uint4* buf;
        unsigned* sink;
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        CUDA_CHECK(cudaMalloc(&sink, 4));
        CUDA_CHECK(cudaMemset(buf, 1, bytes));
        float ms = bench.time([&] { stream_read<<<bench.prop.multiProcessorCount * 16, 256>>>(buf, bytes / 16, sink); }, 5, 30);
        ceiling_GBs = bytes / (ms / 1e3) / 1e9;
        printf("\n[achievable DRAM read bandwidth: stream 256 MB with 16 B loads] %.1f GB/s = %.1f%% of theoretical\n",
               ceiling_GBs, 100.0 * ceiling_GBs / bench.peak_bw_GBs);
        CUDA_CHECK(cudaFree(buf));
        CUDA_CHECK(cudaFree(sink));
    }

    // ---------------- performance ----------------
    struct Perf { int B, L, Hq, Hkv; const char* tag; };
    std::vector<Perf> perf = {
        {1,  32768, 32, 8, "B=1  L=32K (long context)"},
        {8,   8192, 32, 8, "B=8  L=8K"},
        {64,  1024, 32, 8, "B=64 L=1K (large batch)"},
    };
    printf("\n[performance: Hq=32, Hkv=8 (G=4), D=128, fp16, pages scattered]\n");
    printf("  memory-bound kernel: %%BW is vs theoretical %.0f GB/s; achievable ceiling is %.0f GB/s\n",
           bench.peak_bw_GBs, ceiling_GBs);
    bench.header();
    for (auto& c : perf) {
        Batch bt(std::vector<int>(c.B, c.L), c.Hq, c.Hkv, 11, /*fill=*/false);
        DeviceBatch db(bt);
        double bytes = 2.0 * bt.kv_bytes() + 2.0 * bt.q.size() * 2;   // K + V + q + out
        double flops = 4.0 * c.B * c.Hq * (double)c.L * HEAD_DIM;      // QK^T + PV
        printf("  %s\n", c.tag);
        struct Run { int force, v; const char* nm; };
        std::vector<Run> runs = {Run{1, 2, "v2 no split          "},
                                 Run{0, 1, "v1 split-KV          "},
                                 Run{0, 2, "v2 split-KV +swizzle "},
                                 Run{0, 3, "v3 v2 + cp.async x2  "}};
        std::vector<DecodeParams> ps;
        std::vector<std::function<void()>> fns;
        for (auto& r : runs) ps.push_back(db.params(bt, r.force, r.v));
        for (size_t i = 0; i < runs.size(); i++)
            fns.push_back([&, i] { decode_attention(ps[i], c.Hq / c.Hkv, runs[i].v, db.out, db.lse, ws); });
        std::vector<float> tm = bench.time_interleaved(fns, 10, 3, 3, /*cold=*/true);
        for (size_t i = 0; i < runs.size(); i++) {
            char name[96];
            snprintf(name, sizeof name, "  %s(%4d blk)", runs[i].nm, ps[i].num_chunks * c.B * c.Hkv);
            bench.report(name, tm[i], bytes, flops);
        }
    }

    // ---------------- performance: shared-prefix batches ----------------
    struct SPerf { int B, Lp, Ls; const char* tag; };
    std::vector<SPerf> sperf = {
        {64,  4096, 256, "B=64, shared 4K system prompt, 256-token suffixes"},
        {32, 16384, 512, "B=32, shared 16K document,     512-token suffixes"},
        {8,  16384, 512, "B=8,  shared 16K document,     512-token suffixes"},
    };
    printf("\n[performance: shared-prefix batches, Hq=32, Hkv=8, D=128, fp16, L2 flushed before each run]\n");
    printf("  'naive bytes' = KV every request would read without sharing; split-KV v2 reads that\n");
    printf("  through L2, so the L2 (32 MB here) can already dedup part of a small shared prefix.\n");
    printf("  %-52s %10s %10s %12s %9s\n", "", "v2 (ms)", "cascade", "naive GB/s*", "speedup");
    for (auto& c : sperf) {
        Batch bt = make_shared_prefix_batch(c.Lp, std::vector<int>(c.B, c.Ls), 32, 8, 21, false);
        DeviceBatch db(bt);
        DecodeParams p = db.params(bt, 0, 2);
        CascadePlan cp = plan_cascade(p, bt.indptr, bt.indices, bt.lens, db.indices);
        std::vector<float> tm = bench.time_interleaved(
            {[&] { decode_attention(p, 4, 2, db.out, db.lse, ws); }, [&] { cascade_decode(cp, db.out, db.lse, ws); }},
            10, 3, 3, /*cold=*/true);
        float t_v2 = tm[0], t_cas = tm[1];
        double naive = 2.0 * c.B * (double)(c.Lp + c.Ls) * 8 * HEAD_DIM * 2;
        printf("  %-52s %10.3f %10.3f %12.0f %8.2fx\n", c.tag, t_v2, t_cas, naive / t_cas / 1e6, t_v2 / t_cas);
    }
    printf("  * naive bytes / cascade time: the bandwidth a non-sharing kernel would need to match it\n");
    if (ws.part_o) { cudaFree(ws.part_o); cudaFree(ws.part_lse); }
    return bench.finish();
}

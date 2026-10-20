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
#include <numeric>
#include <random>

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
    const int kbeg = chunk * p.chunk_tokens;
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
        size_t row = ((size_t)b * p.Hq + kvh * G + g) * p.num_chunks + chunk;
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
    size_t need = (size_t)p.B * p.Hq * p.num_chunks;
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
    merge_states<<<p.B * p.Hq, HEAD_DIM, 0, s>>>(ws.part_o, ws.part_lse, out, lse, p.num_chunks);
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
        return p;
    }
};

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
        for (Run r : {Run{1, 2, "v2 no split          "},
                      Run{0, 1, "v1 split-KV          "},
                      Run{0, 2, "v2 split-KV +swizzle "},
                      Run{0, 3, "v3 v2 + cp.async x2  "}}) {
            DecodeParams p = db.params(bt, r.force, r.v);
            float ms = bench.time([&] { decode_attention(p, c.Hq / c.Hkv, r.v, db.out, db.lse, ws); }, 5, 30);
            char name[96];
            snprintf(name, sizeof name, "  %s(%4d blk)", r.nm, p.num_chunks * c.B * c.Hkv);
            bench.report(name, ms, bytes, flops);
        }
    }
    if (ws.part_o) { cudaFree(ws.part_o); cudaFree(ws.part_lse); }
    return bench.finish();
}

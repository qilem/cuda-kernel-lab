// 02_moe_grouped_gemm.cu
//
// One MoE expert layer, end to end on the GPU, with no host synchronization:
//
//   router_topk       softmax over E experts per token, pick top-k, renormalize weights
//   moe_count         tokens per expert (histogram)
//   moe_scan          expert row offsets + per-tile-size M-tile offsets
//   moe_permute       copy each (token, slot) row into a buffer sorted by expert
//   moe_grouped_gemm  Yp[rows of e] = Xp[rows of e] @ W[e] for ALL experts in ONE launch
//                     (Tensor Cores via WMMA, cp.async double buffering)
//   moe_combine       out[t] = sum_j w[t,j] * Yp[pos(t,j)]   (gather, fixed order)
//
// The interesting part is the grouped GEMM. Every expert has a different, data-dependent
// row count n_e. The options are:
//   (a) one GEMM launch per expert       -> E launches, and the host must read n_e first
//   (b) pad every expert to max_e n_e    -> one batched launch, but wasted MACs under skew
//   (c) cuBLAS grouped GEMM              -> one call, host still needs n_e for the arrays
//   (d) this kernel: a device-side tile scheduler. The grid is an upper bound on M-tiles;
//       each block binary-searches the expert tile offsets to find its expert and exits
//       if it has no work. Nothing is copied back to the host, so the whole layer can be
//       captured in a CUDA Graph.
//
// Two tile configurations share one templated kernel:
//   small: 64x64 block tile, 16 warps, each warp one 16x16 fragment       (the default)
//   large: 128x128 block tile, 16 warps, each warp 32x32 = 2x2 fragments  (register reuse:
//          each A and B fragment feeds 2 MMAs)
// The large tile was expected to win on big experts (Mixtral-like prefill) and did NOT:
// it was slower on every shape measured, with both 8- and 16-warp layouts. Kept as an
// experiment (row (e) in the output) rather than silently deleted; see README.
//
// Determinism: rows get their slot in the sorted buffer via atomicAdd, so the ORDER of
// rows inside an expert changes from run to run. That is fine: each output row of a GEMM
// depends only on its own input row, and combine is a gather in a fixed j order. Atomics
// are safe for allocating integer slots; they are not safe for accumulating floats
// (moe_combine_atomic is included to show the difference).

#include "common/moe.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <climits>
#include <cstring>
#include <random>
#ifdef _MSC_VER
#pragma comment(lib, "cublas.lib")
#endif
using namespace nvcuda;


template <int BM_, int BN_, int WARPS_M_, int WARPS_N_>
struct TileCfg {
    static constexpr int BM = BM_, BN = BN_, BK = 32;
    static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
    static constexpr int NT  = WARPS_M * WARPS_N * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;     // warp tile
    static constexpr int FM  = WTM / 16, FN = WTN / 16;              // fragments per warp
    static constexpr int LDA = BK + 8;     // +16 B per row: keeps cp.async alignment, skews banks
    static constexpr int LDB = BN + 8;
    static constexpr int A_STAGE = BM * LDA, B_STAGE = BK * LDB;     // halves per stage
    static constexpr int PIPE_BYTES = 2 * (A_STAGE + B_STAGE) * 2;   // double buffered
    static constexpr int STG_LD = 20;                                 // per-warp 16x16 fp32 staging
    static constexpr int STG_BYTES = WARPS_M * WARPS_N * 16 * STG_LD * 4;
    static constexpr int SMEM = PIPE_BYTES > STG_BYTES ? PIPE_BYTES : STG_BYTES;
};
using SmallTile = TileCfg<64, 64, 4, 4>;
using LargeTile = TileCfg<128, 128, 4, 4>;

// E <= 256, so a serial scan by one thread costs a few microseconds and keeps this simple.
// Tile offsets are produced for both tile heights so either GEMM config can run.
__global__ void moe_scan(const int* counts, int* offsets, int* tiles_s, int* tiles_l, int* cursor, int E) {
    if (threadIdx.x != 0) return;
    int o = 0, ts = 0, tl = 0;
    for (int e = 0; e < E; e++) {
        offsets[e] = o; tiles_s[e] = ts; tiles_l[e] = tl;
        o  += counts[e];
        ts += (counts[e] + SmallTile::BM - 1) / SmallTile::BM;
        tl += (counts[e] + LargeTile::BM - 1) / LargeTile::BM;
        cursor[e] = 0;
    }
    offsets[E] = o; tiles_s[E] = ts; tiles_l[E] = tl;
}

// ---------------------------------------------------------------- grouped GEMM
// grid = (upper bound on total M-tiles, f / BN)
template <class C>
__global__ void __launch_bounds__(C::NT)
moe_grouped_gemm(const half* Xp, const half* W, half* Yp,
                 const int* offsets, const int* tile_offsets, int E, int d, int f) {
    __shared__ __align__(128) unsigned char smem[C::SMEM];
    half* As = reinterpret_cast<half*>(smem);               // [2][BM][LDA]
    half* Bs = As + 2 * C::A_STAGE;                         // [2][BK][LDB]

    const int mt = blockIdx.x;
    if (mt >= tile_offsets[E]) return;                      // surplus block from the upper bound

    int lo = 0, hi = E;                                     // largest e with tile_offsets[e] <= mt
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (tile_offsets[mid] <= mt) lo = mid; else hi = mid;
    }
    const int e       = lo;
    const int row0    = offsets[e] + (mt - tile_offsets[e]) * C::BM;
    const int rows    = min(C::BM, offsets[e + 1] - row0);
    const int colBase = blockIdx.y * C::BN;
    const half* B = W + (size_t)e * d * f;

    const int tid = threadIdx.x, wid = tid / 32, lane = tid % 32;
    const int wm = wid / C::WARPS_N, wn = wid % C::WARPS_N;

    auto load_stage = [&](int s, int k0) {
        half* as = As + s * C::A_STAGE;
        half* bs = Bs + s * C::B_STAGE;
        for (int i = tid; i < C::BM * (C::BK / 8); i += C::NT) {
            int r = i / (C::BK / 8), c = (i % (C::BK / 8)) * 8;
            int src = row0 + min(r, rows - 1);              // tail rows re-read a valid row and
            __pipeline_memcpy_async(as + r * C::LDA + c, Xp + (size_t)src * d + k0 + c, 16);  // are never stored
        }
        for (int i = tid; i < C::BK * (C::BN / 8); i += C::NT) {
            int r = i / (C::BN / 8), c = (i % (C::BN / 8)) * 8;
            __pipeline_memcpy_async(bs + r * C::LDB + c, B + (size_t)(k0 + r) * f + colBase + c, 16);
        }
        __pipeline_commit();
    };

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[C::FM];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[C::FN];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[C::FM][C::FN];
    #pragma unroll
    for (int i = 0; i < C::FM; i++)
        #pragma unroll
        for (int j = 0; j < C::FN; j++) wmma::fill_fragment(acc[i][j], 0.f);

    const int nt = d / C::BK;
    load_stage(0, 0);
    int s = 0;
    for (int t = 0; t < nt; t++) {
        if (t + 1 < nt) {
            load_stage(s ^ 1, (t + 1) * C::BK);
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        const half* as = As + s * C::A_STAGE;
        const half* bs = Bs + s * C::B_STAGE;
        #pragma unroll
        for (int kk = 0; kk < C::BK; kk += 16) {
            #pragma unroll
            for (int i = 0; i < C::FM; i++)
                wmma::load_matrix_sync(a[i], as + (wm * C::WTM + i * 16) * C::LDA + kk, C::LDA);
            #pragma unroll
            for (int j = 0; j < C::FN; j++)
                wmma::load_matrix_sync(b[j], bs + kk * C::LDB + wn * C::WTN + j * 16, C::LDB);
            #pragma unroll
            for (int i = 0; i < C::FM; i++)
                #pragma unroll
                for (int j = 0; j < C::FN; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
        s ^= 1;
    }

    // Masked epilogue: rows past this expert's end belong to the NEXT expert in Yp, so a
    // direct store_matrix_sync to global would race with another block. Each warp stages one
    // 16x16 fragment at a time in (now free) shared memory and writes only the valid rows.
    float* stg = reinterpret_cast<float*>(smem) + wid * 16 * C::STG_LD;
    const int lr = lane / 2, lc = (lane % 2) * 8;
    #pragma unroll
    for (int i = 0; i < C::FM; i++)
        #pragma unroll
        for (int j = 0; j < C::FN; j++) {
            wmma::store_matrix_sync(stg, acc[i][j], C::STG_LD, wmma::mem_row_major);
            __syncwarp();
            int r = wm * C::WTM + i * 16 + lr;
            int c = wn * C::WTN + j * 16 + lc;
            if (r < rows) {
                __align__(16) half2 h[4];
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    h[q] = __floats2half2_rn(stg[lr * C::STG_LD + lc + 2 * q], stg[lr * C::STG_LD + lc + 2 * q + 1]);
                *reinterpret_cast<uint4*>(Yp + (size_t)(row0 + r) * f + colBase + c) = *reinterpret_cast<uint4*>(h);
            }
            __syncwarp();
        }
}

// ---------------------------------------------------------------- helpers
__global__ void pad_experts(const half* Xp, const int* offsets, half* Xpad, int max_n, int d) {
    const int e = blockIdx.y, r = blockIdx.x;
    const int n_e = offsets[e + 1] - offsets[e];
    uint4* dst = reinterpret_cast<uint4*>(Xpad + ((size_t)e * max_n + r) * d);
    const uint4* src = reinterpret_cast<const uint4*>(Xp + (size_t)(offsets[e] + r) * d);
    for (int c = threadIdx.x; c < d / 8; c += blockDim.x)
        dst[c] = (r < n_e) ? src[c] : make_uint4(0, 0, 0, 0);
}

enum TileChoice { AUTO = 0, SMALL = 1, LARGE = 2 };

struct Moe {
    int T, E, k, d, f;
    half *X, *W, *Xp, *Yp, *out;
    float *logits, *wts;
    int *ids, *counts, *offsets, *tiles_s, *tiles_l, *cursor, *inv_perm;

    Moe(int T_, int E_, int k_, int d_, int f_) : T(T_), E(E_), k(k_), d(d_), f(f_) {
        if (E > MAX_E || k > MAX_K || k > E || d % 32 || f % SmallTile::BN) {
            fprintf(stderr, "unsupported MoE shape\n");
            exit(1);
        }
        size_t A = (size_t)T * k;
        CUDA_CHECK(cudaMalloc(&X, (size_t)T * d * 2));
        CUDA_CHECK(cudaMalloc(&W, (size_t)E * d * f * 2));
        CUDA_CHECK(cudaMalloc(&Xp, A * d * 2));
        CUDA_CHECK(cudaMalloc(&Yp, A * f * 2));
        CUDA_CHECK(cudaMalloc(&out, (size_t)T * f * 2));
        CUDA_CHECK(cudaMalloc(&logits, (size_t)T * E * 4));
        CUDA_CHECK(cudaMalloc(&wts, A * 4));
        CUDA_CHECK(cudaMalloc(&ids, A * 4));
        CUDA_CHECK(cudaMalloc(&inv_perm, A * 4));
        CUDA_CHECK(cudaMalloc(&counts, E * 4));
        CUDA_CHECK(cudaMalloc(&cursor, E * 4));
        CUDA_CHECK(cudaMalloc(&offsets, (E + 1) * 4));
        CUDA_CHECK(cudaMalloc(&tiles_s, (E + 1) * 4));
        CUDA_CHECK(cudaMalloc(&tiles_l, (E + 1) * 4));
    }
    ~Moe() {
        cudaFree(X); cudaFree(W); cudaFree(Xp); cudaFree(Yp); cudaFree(out);
        cudaFree(logits); cudaFree(wts); cudaFree(ids); cudaFree(inv_perm);
        cudaFree(counts); cudaFree(cursor); cudaFree(offsets); cudaFree(tiles_s); cudaFree(tiles_l);
    }
    // The measured winner on every shape so far is the 64x64 config (see header comment).
    TileChoice pick() const { return SMALL; }
    // sum_e ceil(n_e / BM) <= ceil(A / BM) + min(E, A): a launch bound that needs no n_e.
    int tile_bound(int bm) const { int A = T * k; return ceil_div(A, bm) + std::min(E, A); }

    void route_and_permute(cudaStream_t s = 0) {
        int A = T * k;
        router_topk<<<ceil_div(T * 32, 256), 256, 0, s>>>(logits, ids, wts, T, E, k);
        CUDA_CHECK(cudaMemsetAsync(counts, 0, E * 4, s));
        moe_count<<<ceil_div(A, 256), 256, 0, s>>>(ids, counts, A);
        moe_scan<<<1, 32, 0, s>>>(counts, offsets, tiles_s, tiles_l, cursor, E);
        moe_permute<<<A, 128, 0, s>>>(X, ids, offsets, cursor, inv_perm, Xp, k, d);
    }
    void grouped_gemm(TileChoice c = AUTO, cudaStream_t s = 0) {
        if (c == AUTO) c = pick();
        if (c == LARGE) {
            dim3 g(tile_bound(LargeTile::BM), f / LargeTile::BN);
            moe_grouped_gemm<LargeTile><<<g, LargeTile::NT, 0, s>>>(Xp, W, Yp, offsets, tiles_l, E, d, f);
        } else {
            dim3 g(tile_bound(SmallTile::BM), f / SmallTile::BN);
            moe_grouped_gemm<SmallTile><<<g, SmallTile::NT, 0, s>>>(Xp, W, Yp, offsets, tiles_s, E, d, f);
        }
    }
    void combine(cudaStream_t s = 0) { moe_combine<<<T, 256, 0, s>>>(Yp, inv_perm, wts, out, k, f); }
    void forward(TileChoice c = AUTO, cudaStream_t s = 0) { route_and_permute(s); grouped_gemm(c, s); combine(s); }
};

void fill_inputs(Moe& m, const std::vector<float>& lg, unsigned seed) {
    CUDA_CHECK(cudaMemcpy(m.logits, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
    init_uniform<<<1024, 256>>>(m.X, (size_t)m.T * m.d, seed, 1.0f);
    init_uniform<<<1024, 256>>>(m.W, (size_t)m.E * m.d * m.f, seed + 1, 1.0f / std::sqrt((float)m.d));
    CUDA_CHECK_KERNEL();
}

// fp64 reference: routing + per-expert GEMM + weighted sum, from the definition.
void cpu_reference(const Moe& m, const std::vector<float>& lg, std::vector<float>& out,
                   std::vector<int>& ref_ids) {
    std::vector<float> X = to_float(m.X, (size_t)m.T * m.d);
    std::vector<float> W = to_float(m.W, (size_t)m.E * m.d * m.f);
    out.assign((size_t)m.T * m.f, 0.f);
    ref_ids.assign((size_t)m.T * m.k, -1);
    for (int t = 0; t < m.T; t++) {
        std::vector<double> p(m.E);
        double mx = -1e300, s = 0;
        for (int e = 0; e < m.E; e++) mx = std::max(mx, (double)lg[(size_t)t * m.E + e]);
        for (int e = 0; e < m.E; e++) { p[e] = std::exp(lg[(size_t)t * m.E + e] - mx); s += p[e]; }
        std::vector<bool> used(m.E, false);
        std::vector<int> sel;
        std::vector<double> sw;
        double tot = 0;
        for (int r = 0; r < m.k; r++) {
            int bi = -1;
            for (int e = 0; e < m.E; e++) if (!used[e] && (bi < 0 || p[e] > p[bi])) bi = e;
            used[bi] = true;
            sel.push_back(bi);
            sw.push_back(p[bi] / s);
            tot += p[bi] / s;
            ref_ids[(size_t)t * m.k + r] = bi;
        }
        for (int r = 0; r < m.k; r++) {
            const float* We = &W[(size_t)sel[r] * m.d * m.f];
            for (int j = 0; j < m.f; j++) {
                double acc = 0;
                for (int c = 0; c < m.d; c++) acc += (double)X[(size_t)t * m.d + c] * We[(size_t)c * m.f + j];
                out[(size_t)t * m.f + j] += (float)(sw[r] / tot * acc);
            }
        }
    }
}

unsigned bits(float x) { unsigned u; memcpy(&u, &x, 4); return u; }

int main() {
    Bench bench;
    cublasHandle_t h;
    cublasCreate(&h);

    // ================= correctness =================
    struct SmallCase { int T, E, k, d, f, disable; const char* tag; };
    std::vector<SmallCase> small = {
        {100,  8, 2, 256, 384, -1, "E=8  k=2  (Mixtral-style)"},
        { 37, 16, 4, 128, 256,  3, "E=16 k=4, one expert empty"},
        { 64, 64, 6, 128, 128, -1, "E=64 k=6  (fine-grained)"},
    };
    printf("\n[correctness: full MoE layer vs fp64 reference, both tile configs]\n");
    for (auto& c : small) {
        Moe m(c.T, c.E, c.k, c.d, c.f);
        auto lg = make_logits(c.T, c.E, 5, c.disable);
        fill_inputs(m, lg, 21);
        std::vector<float> ref;
        std::vector<int> ref_ids, got_ids((size_t)c.T * c.k);
        cpu_reference(m, lg, ref, ref_ids);
        for (TileChoice tc : {SMALL, LARGE}) {
            m.forward(tc);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(got_ids.data(), m.ids, got_ids.size() * 4, cudaMemcpyDeviceToHost));
            int id_bad = 0;
            for (size_t i = 0; i < got_ids.size(); i++) id_bad += (got_ids[i] != ref_ids[i]);
            std::vector<float> got = to_float(m.out, (size_t)c.T * c.f);
            char tag[112];
            snprintf(tag, sizeof tag, "%-28s %s tiles, routing mismatches=%d", c.tag,
                     tc == SMALL ? "64x64  " : "128x128", id_bad);
            bench.verify(got.data(), ref.data(), got.size(), tag, 1e-2f, 5e-3f);
            if (id_bad) bench.n_fail++;
        }
    }

    // ================= performance =================
    struct Cfg { const char* name; int T, E, k, d, f; };
    std::vector<Cfg> cfgs = {
        {"Mixtral-8x7B-like  prefill  T=512  E=8  k=2 d=4096 f=14336", 512, 8, 2, 4096, 14336},
        {"fine-grained MoE   prefill  T=512  E=64 k=6 d=2048 f=1408 ", 512, 64, 6, 2048, 1408},
        {"fine-grained MoE   decode   T=16   E=64 k=6 d=2048 f=1408 ", 16, 64, 6, 2048, 1408},
    };
    for (auto& c : cfgs) {
        printf("\n=====================================================================\n");
        printf(" %s\n", c.name);
        Moe m(c.T, c.E, c.k, c.d, c.f);
        auto lg = make_logits(c.T, c.E, 9);
        fill_inputs(m, lg, 33);
        m.route_and_permute();
        CUDA_CHECK_KERNEL();

        std::vector<int> counts(c.E), offs(c.E + 1);
        CUDA_CHECK(cudaMemcpy(counts.data(), m.counts, c.E * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(offs.data(), m.offsets, (c.E + 1) * 4, cudaMemcpyDeviceToHost));
        int A = c.T * c.k, max_n = 0, min_n = INT_MAX, active = 0;
        for (int n : counts) { max_n = std::max(max_n, n); min_n = std::min(min_n, n); active += (n > 0); }
        printf(" tokens/expert: min %d  mean %.1f  max %d   (%d of %d experts active)  -> auto picks %s tiles\n",
               min_n, (double)A / c.E, max_n, active, c.E, m.pick() == LARGE ? "128x128" : "64x64");

        double useful = 2.0 * A * (double)c.d * c.f;
        double wbytes = (double)active * c.d * c.f * 2;     // weights that must be streamed
        size_t ybytes = (size_t)A * c.f * 2;

        half* Yref;
        CUDA_CHECK(cudaMalloc(&Yref, ybytes));
        const float al = 1.f, be = 0.f;

        // (a) one cuBLAS GEMM per expert (host already knows n_e here; that read is not timed)
        auto per_expert = [&] {
            for (int e = 0; e < c.E; e++) {
                if (!counts[e]) continue;
                cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, c.f, counts[e], c.d, &al,
                             m.W + (size_t)e * c.d * c.f, CUDA_R_16F, c.f,
                             m.Xp + (size_t)offs[e] * c.d, CUDA_R_16F, c.d, &be,
                             Yref + (size_t)offs[e] * c.f, CUDA_R_16F, c.f,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
        };

        // (b) pad every expert to max_n, one strided-batched GEMM (padding copy not timed)
        half *Xpad, *Ypad;
        CUDA_CHECK(cudaMalloc(&Xpad, (size_t)c.E * max_n * c.d * 2));
        CUDA_CHECK(cudaMalloc(&Ypad, (size_t)c.E * max_n * c.f * 2));
        pad_experts<<<dim3(max_n, c.E), 128>>>(m.Xp, m.offsets, Xpad, max_n, c.d);
        CUDA_CHECK_KERNEL();
        auto padded = [&] {
            cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, c.f, max_n, c.d, &al,
                                       m.W, CUDA_R_16F, c.f, (long long)c.d * c.f,
                                       Xpad, CUDA_R_16F, c.d, (long long)max_n * c.d, &be,
                                       Ypad, CUDA_R_16F, c.f, (long long)max_n * c.f, c.E,
                                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };

        // (c) cuBLAS grouped GEMM, one group per non-empty expert (array setup not timed)
        std::vector<cublasOperation_t> ta, tb;
        std::vector<int> gm, gn, gk, glda, gldb, gldc, gsize;
        std::vector<float> galpha, gbeta;
        std::vector<const void*> pA, pB;
        std::vector<void*> pC;
        for (int e = 0; e < c.E; e++) {
            if (!counts[e]) continue;
            ta.push_back(CUBLAS_OP_N); tb.push_back(CUBLAS_OP_N);
            gm.push_back(c.f); gn.push_back(counts[e]); gk.push_back(c.d);
            glda.push_back(c.f); gldb.push_back(c.d); gldc.push_back(c.f);
            galpha.push_back(1.f); gbeta.push_back(0.f); gsize.push_back(1);
            pA.push_back(m.W + (size_t)e * c.d * c.f);
            pB.push_back(m.Xp + (size_t)offs[e] * c.d);
            pC.push_back(Yref + (size_t)offs[e] * c.f);
        }
        int G = (int)gsize.size();
        void **dA, **dB, **dC;
        CUDA_CHECK(cudaMalloc(&dA, G * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&dB, G * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&dC, G * sizeof(void*)));
        CUDA_CHECK(cudaMemcpy(dA, pA.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, pB.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, pC.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
        cublasStatus_t gst = CUBLAS_STATUS_SUCCESS;
        auto grouped = [&] {
            gst = cublasGemmGroupedBatchedEx(h, ta.data(), tb.data(), gm.data(), gn.data(), gk.data(),
                                             galpha.data(), (const void* const*)dA, CUDA_R_16F, glda.data(),
                                             (const void* const*)dB, CUDA_R_16F, gldb.data(),
                                             gbeta.data(), (void* const*)dC, CUDA_R_16F, gldc.data(),
                                             G, gsize.data(), CUBLAS_COMPUTE_32F);
        };

        // correctness of both grouped-kernel configs on the big shape vs per-expert cuBLAS
        per_expert();
        CUDA_CHECK_KERNEL();
        std::vector<float> ref = to_float(Yref, (size_t)A * c.f);
        for (TileChoice tc : {SMALL, LARGE}) {
            CUDA_CHECK(cudaMemset(m.Yp, 0, ybytes));
            m.grouped_gemm(tc);
            CUDA_CHECK_KERNEL();
            std::vector<float> got = to_float(m.Yp, (size_t)A * c.f);
            bench.verify(got.data(), ref.data(), got.size(),
                         tc == SMALL ? "grouped GEMM 64x64   vs per-expert cuBLAS"
                                     : "grouped GEMM 128x128 vs per-expert cuBLAS", 1e-2f, 2e-3f);
        }

        grouped();
        CUDA_CHECK_KERNEL();
        bool have_c = (gst == CUBLAS_STATUS_SUCCESS);
        std::vector<std::function<void()>> fns = {per_expert, padded,
                                                  [&] { m.grouped_gemm(SMALL); }, [&] { m.grouped_gemm(LARGE); }};
        if (have_c) fns.push_back(grouped);
        std::vector<float> tm = bench.time_interleaved(fns);   // all methods see the same thermal state
        float t_a = tm[0], t_b = tm[1], t_s = tm[2], t_l = tm[3];
        float t_c = have_c ? tm[4] : -1.f;

        double padded_flops = 2.0 * c.E * max_n * (double)c.d * c.f;
        printf("\n expert GEMM stage (%.1f useful GFLOP, %.0f MB of active expert weights)\n",
               useful / 1e9, wbytes / 1e6);
        printf("   %-40s %9s %9s %11s %8s\n", "method", "time(ms)", "TFLOP/s*", "weight GB/s", "launches");
        auto row = [&](const char* nm, float ms, int launches) {
            if (ms < 0) { printf("   %-40s   not supported for this type combo (status %d)\n", nm, (int)gst); return; }
            printf("   %-40s %9.3f %9.1f %11.1f %8d\n", nm, ms, useful / ms / 1e9, wbytes / ms / 1e6, launches);
        };
        row("(a) cuBLAS, one GEMM per expert", t_a, active);
        char pad_nm[64];
        snprintf(pad_nm, sizeof pad_nm, "(b) cuBLAS batched, padded (%.1fx MACs)", padded_flops / useful);
        row(pad_nm, t_b, 1);
        row("(c) cuBLAS grouped GEMM", t_c, 1);
        row("(d) this repo, 64x64 tiles", t_s, 1);
        row("(e) this repo, 128x128 tiles", t_l, 1);
        printf("   * useful FLOPs only (padding work is not credited)\n");

        // full layer (auto tile choice), plus determinism of gather vs atomic combine
        float t_full = bench.time([&] { m.forward(); }, 3, 20);
        float t_comb = bench.time([&] { m.combine(); }, 3, 20);
        float* outf;
        CUDA_CHECK(cudaMalloc(&outf, (size_t)c.T * c.f * 4));
        auto atomic_combine = [&] {
            CUDA_CHECK(cudaMemsetAsync(outf, 0, (size_t)c.T * c.f * 4));
            moe_combine_atomic<<<A, 256>>>(m.Yp, m.inv_perm, m.wts, outf, c.k, c.f);
        };
        float t_comb_at = bench.time(atomic_combine, 3, 20);
        float t_gemm = (m.pick() == LARGE) ? t_l : t_s;
        printf("\n full layer, auto tiles (route + permute + grouped GEMM + combine): %.3f ms  (GEMM = %.0f%%)\n",
               t_full, 100.0 * t_gemm / t_full);

        std::vector<half> o1((size_t)c.T * c.f), o2(o1.size());
        m.forward(); CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(o1.data(), m.out, o1.size() * 2, cudaMemcpyDeviceToHost));
        m.forward(); CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(o2.data(), m.out, o2.size() * 2, cudaMemcpyDeviceToHost));
        size_t diff_g = 0;
        for (size_t i = 0; i < o1.size(); i++)
            diff_g += (memcmp(&o1[i], &o2[i], 2) != 0);

        std::vector<float> f1((size_t)c.T * c.f), f2(f1.size());
        atomic_combine(); CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(f1.data(), outf, f1.size() * 4, cudaMemcpyDeviceToHost));
        atomic_combine(); CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(f2.data(), outf, f2.size() * 4, cudaMemcpyDeviceToHost));
        size_t diff_a = 0;
        for (size_t i = 0; i < f1.size(); i++) diff_a += (bits(f1[i]) != bits(f2[i]));
        printf(" combine, gather (fixed order): %.3f ms, outputs differing across 2 full runs: %zu / %zu\n",
               t_comb, diff_g, o1.size());
        printf(" combine, atomicAdd scatter   : %.3f ms, outputs differing across 2 runs     : %zu / %zu\n",
               t_comb_at, diff_a, f1.size());
        if (diff_g) bench.n_fail++;

        CUDA_CHECK(cudaFree(outf)); CUDA_CHECK(cudaFree(Yref));
        CUDA_CHECK(cudaFree(Xpad)); CUDA_CHECK(cudaFree(Ypad));
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }
    cublasDestroy(h);
    return bench.finish();
}

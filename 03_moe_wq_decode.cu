// 03_moe_wq_decode.cu
//
// Weight-only quantized MoE expert GEMM for decode: W8A16 and W4A16 (fp16 activations).
//
// In decode a MoE layer sees only a handful of tokens per expert, so the expert GEMM is
// bounded by how fast expert weights stream from DRAM. 02 measured both cuBLAS grouped GEMM
// and our fp16 grouped GEMM at ~330 GB/s against a ~342 GB/s ceiling: at fp16 there is
// nothing left to win. The only way to go faster is to read fewer bytes. cuBLAS grouped GEMM
// has no weight-only quantized path, which is what this file adds and measures.
//
// Quantization: symmetric, one fp16 scale per (output column, group of 128 along K).
// Unlike the INT8 x INT8 GEMM in 07, scales may vary along K here, because weights are
// dequantized to fp32 inside the K loop instead of being accumulated as integers.
//
// Kernel (GEMV-shaped, because decode gives few rows per expert):
//   block  = (expert e, 32*CPT consecutive output columns); experts with no tokens exit
//   warps  = 4, each takes 1/4 of K; partial sums reduced through shared memory at the end
//   thread = CPT columns; one 32-bit word per k (2 fp16 / 4 int8 / 8 int4 weights), packed
//            so that one 16-byte load brings 4 consecutive k
//   rows   = processed in passes of R in {1, 2, 4, 8}, templated so accumulators stay in
//            registers (decode averages 1-2 rows per expert)
//
// Prior work: Marlin, vLLM fused_marlin_moe, TensorRT-LLM weight-only GEMM. Not new.

#include "common/moe.cuh"
#include <cublas_v2.h>
#ifdef _MSC_VER
#pragma comment(lib, "cublas.lib")
#endif

constexpr int GS   = 128;          // quantization group size along K
constexpr int NW   = 4;            // warps per block (split K)
constexpr int NTW  = NW * 32;
constexpr int RMAX = 8;            // max rows per pass

template <int BITS> struct WQ {
    static constexpr int CPT  = 32 / BITS;          // columns per thread
    static constexpr int COLS = 32 * CPT;           // columns per block
    static constexpr int QMAX = (BITS == 8) ? 127 : 7;
};

// 32-bit word holding (k, column-word cw) of expert e; 4 consecutive k are adjacent.
__host__ __device__ inline size_t widx(int e, int k, int cw, int d, int ncw) {
    return (((size_t)e * (d / 4) + k / 4) * ncw + cw) * 4 + (k % 4);
}

// fp16 W [E][d][f] -> packed words (+ per-group scales for BITS < 16).
template <int BITS>
__global__ void pack_weights(const half* W, uint32_t* P, half* Sc, int E, int d, int f) {
    using C = WQ<BITS>;
    const int ncw = f / C::CPT, ngr = d / GS;
    const size_t total = (size_t)E * ngr * ncw;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        int cw = (int)(i % ncw);
        int g  = (int)((i / ncw) % ngr);
        int e  = (int)(i / ((size_t)ncw * ngr));
        int c0 = cw * C::CPT;
        const half* We = W + (size_t)e * d * f;
        if constexpr (BITS == 16) {
            for (int k = g * GS; k < (g + 1) * GS; k++) {
                uint32_t lo = __half_as_ushort(We[(size_t)k * f + c0]);
                uint32_t hi = __half_as_ushort(We[(size_t)k * f + c0 + 1]);
                P[widx(e, k, cw, d, ncw)] = lo | (hi << 16);
            }
        } else {
            float s[C::CPT];
            for (int c = 0; c < C::CPT; c++) {
                float m = 0.f;
                for (int k = g * GS; k < (g + 1) * GS; k++) m = fmaxf(m, fabsf(__half2float(We[(size_t)k * f + c0 + c])));
                half hs = __float2half(m > 0.f ? m / C::QMAX : 1.f);
                Sc[((size_t)e * ngr + g) * f + c0 + c] = hs;
                s[c] = __half2float(hs);                       // quantize with the rounded scale
            }
            for (int k = g * GS; k < (g + 1) * GS; k++) {
                uint32_t word = 0;
                for (int c = 0; c < C::CPT; c++) {
                    int q = __float2int_rn(__half2float(We[(size_t)k * f + c0 + c]) / s[c]);
                    q = max(-C::QMAX, min(C::QMAX, q));
                    // offset-binary (q + 128 / q + 8) so the fp16 magic-number trick can decode it
                    if constexpr (BITS == 8) word |= (uint32_t)(q + 128) << (8 * c);
                    else                     word |= (uint32_t)(q + 8) << (4 * c);
                }
                P[widx(e, k, cw, d, ncw)] = word;
            }
        }
    }
}

// Unscaled integer weights of one 32-bit word as floats, column c -> out[c].
// fp16 magic number: OR a value v < 1024 into the mantissa of 0x6400 (= 1024.0) and the half
// equals 1024 + v exactly; one more half2 subtract removes 1024 and the offset-binary bias.
// That decodes two weights with one bitwise op and one HSUB2, instead of a shift, mask,
// int->float convert and subtract per weight.
template <int BITS>
__device__ __forceinline__ void decode_word(uint32_t w, float* out) {
    if constexpr (BITS == 16) {
        float2 v = __half22float2(*reinterpret_cast<const __half2*>(&w));
        out[0] = v.x;
        out[1] = v.y;
    } else if constexpr (BITS == 8) {
        const __half2 bias = __float2half2_rn(1024.f + 128.f);
        #pragma unroll
        for (int p = 0; p < 2; p++) {                       // pair p holds columns p and p + 2
            uint32_t h = ((w >> (8 * p)) & 0x00FF00FFu) | 0x64006400u;
            float2 v = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&h), bias));
            out[p] = v.x;
            out[p + 2] = v.y;
        }
    } else {
        const __half2 bias = __float2half2_rn(1024.f + 8.f);
        #pragma unroll
        for (int p = 0; p < 4; p++) {                       // pair p holds columns p and p + 4
            uint32_t h = ((w >> (4 * p)) & 0x000F000Fu) | 0x64006400u;
            float2 v = __half22float2(__hsub2(*reinterpret_cast<__half2*>(&h), bias));
            out[p] = v.x;
            out[p + 4] = v.y;
        }
    }
}

// One pass over R rows of expert e: stream this block's weight columns over K range
// [kbase, kbase + klen) once. part == nullptr: write fp16 output; otherwise write the fp32
// partial sum of this K range (split-K across blocks, reduced by moe_splitk_reduce).
template <int BITS, int R>
__device__ __forceinline__ void gemv_pass(const half* Xp, const uint4* P4, const half* Sc, half* Yp,
                                          float* part, half (*xs)[GS], float* red, int e, int row0, int rows,
                                          int kbase, int klen, int d, int f, int ncw, int cw, bool col_ok,
                                          int warp, int lane) {
    using C = WQ<BITS>;
    float acc[R][C::CPT];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int c = 0; c < C::CPT; c++) acc[r][c] = 0.f;

    const int kslice = klen / NW, kb = kbase + warp * kslice;
    for (int kg = kb; kg < kb + kslice; kg += GS) {
        // Issue LIF independent 16-byte weight loads before consuming any of them. With only
        // 1-2 rows of work per weight, a thread computes far faster than DRAM answers, so
        // throughput is set by how many loads each thread keeps in flight (Little's law).
        // The first batch goes out BEFORE this group's activation and scale loads, so their
        // latency overlaps with the weights instead of stalling the start of every group.
        const uint4* pw = P4 + ((size_t)e * (d / 4) + kg / 4) * ncw + cw;
        constexpr int LIF = 8;                                  // loads in flight per thread
        uint4 w4[LIF];
        #pragma unroll
        for (int u = 0; u < LIF; u++)
            w4[u] = col_ok ? __ldg(pw + (size_t)u * ncw) : make_uint4(0, 0, 0, 0);

        // CPT consecutive fp16 scales = 8 B (int8) or 16 B (int4): one vector load, used at the end
        __align__(16) half sh[C::CPT];
        (void)sh;                                           // unused in the fp16 instantiation
        if constexpr (BITS < 16) {
            const half* sp = Sc + ((size_t)e * (d / GS) + kg / GS) * f + cw * C::CPT;
            if (col_ok) {
                if constexpr (C::CPT == 8) *reinterpret_cast<uint4*>(sh) = *reinterpret_cast<const uint4*>(sp);
                else                       *reinterpret_cast<uint2*>(sh) = *reinterpret_cast<const uint2*>(sp);
            }
        }

        // stage this warp's activation slice [rows] x [kg, kg+GS) in shared memory (broadcast reads)
        for (int i = lane; i < R * (GS / 8); i += 32) {
            int r = i / (GS / 8), c8 = (i % (GS / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < rows) v = *reinterpret_cast<const uint4*>(Xp + (size_t)(row0 + r) * d + kg + c8);
            *reinterpret_cast<uint4*>(&xs[r][c8]) = v;
        }
        __syncwarp();

        // Accumulate x * q with the raw integer q over the whole group, then apply the
        // group's scale once per column, instead of multiplying every weight by its scale.
        float accg[R][C::CPT];
        #pragma unroll
        for (int r = 0; r < R; r++)
            #pragma unroll
            for (int c = 0; c < C::CPT; c++) accg[r][c] = 0.f;

        #pragma unroll 1
        for (int kk0 = 0; kk0 < GS; kk0 += 4 * LIF) {
            if (kk0 > 0) {
                #pragma unroll
                for (int u = 0; u < LIF; u++)
                    w4[u] = col_ok ? __ldg(pw + (size_t)(kk0 / 4 + u) * ncw) : make_uint4(0, 0, 0, 0);
            }
            #pragma unroll
            for (int u = 0; u < LIF; u++) {
                const uint32_t ws[4] = {w4[u].x, w4[u].y, w4[u].z, w4[u].w};
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float q[C::CPT];
                    decode_word<BITS>(ws[j], q);
                    #pragma unroll
                    for (int r = 0; r < R; r++) {
                        float x = __half2float(xs[r][kk0 + 4 * u + j]);
                        #pragma unroll
                        for (int c = 0; c < C::CPT; c++) accg[r][c] = fmaf(x, q[c], accg[r][c]);
                    }
                }
            }
        }

        float s[C::CPT];
        #pragma unroll
        for (int c = 0; c < C::CPT; c++) {
            if constexpr (BITS == 16) s[c] = 1.f;
            else                      s[c] = col_ok ? __half2float(sh[c]) : 0.f;
        }
        #pragma unroll
        for (int r = 0; r < R; r++)
            #pragma unroll
            for (int c = 0; c < C::CPT; c++) acc[r][c] = fmaf(s[c], accg[r][c], acc[r][c]);
        __syncwarp();
    }

    // reduce the 4 K-slices
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int c = 0; c < C::CPT; c++) red[(warp * RMAX + r) * C::COLS + lane * C::CPT + c] = acc[r][c];
    __syncthreads();
    for (int i = threadIdx.x; i < R * C::COLS; i += NTW) {
        int r = i / C::COLS, col = i % C::COLS;
        int gcol = blockIdx.x * C::COLS + col;
        if (r < rows && gcol < f) {
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < NW; w++) sum += red[(w * RMAX + r) * C::COLS + col];
            if (part) part[(size_t)(row0 + r) * f + gcol] = sum;
            else      Yp[(size_t)(row0 + r) * f + gcol] = __float2half(sum);
        }
    }
    __syncthreads();
}

// grid = (ceil(ncw / 32), E, ksplit). With ksplit > 1 each block covers d / ksplit of K and
// writes fp32 partials to ws[kz][row][col]; A = total rows (stride between K-split slices).
template <int BITS>
__global__ void __launch_bounds__(NTW)
moe_wq_gemv(const half* Xp, const uint32_t* P, const half* Sc, half* Yp, const int* offsets,
            int d, int f, int ksplit, float* ws, int A) {
    using C = WQ<BITS>;
    __shared__ __align__(16) half xs[NW][RMAX][GS];
    __shared__ float red[NW * RMAX * C::COLS];
    const int e = blockIdx.y;
    const int base = offsets[e], n = offsets[e + 1] - base;
    if (n == 0) return;                                        // expert received no tokens
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ncw = f / C::CPT, cw = blockIdx.x * 32 + lane;
    const bool col_ok = cw < ncw;
    const uint4* P4 = reinterpret_cast<const uint4*>(P);
    const int klen = d / ksplit, kbase = blockIdx.z * klen;
    float* part = (ksplit > 1) ? ws + (size_t)blockIdx.z * A * f : nullptr;
    for (int r0 = 0; r0 < n; r0 += RMAX) {                     // n is uniform across the block
        int rows = min(RMAX, n - r0);
        if      (rows > 4) gemv_pass<BITS, 8>(Xp, P4, Sc, Yp, part, xs[warp], red, e, base + r0, rows, kbase, klen, d, f, ncw, cw, col_ok, warp, lane);
        else if (rows > 2) gemv_pass<BITS, 4>(Xp, P4, Sc, Yp, part, xs[warp], red, e, base + r0, rows, kbase, klen, d, f, ncw, cw, col_ok, warp, lane);
        else if (rows > 1) gemv_pass<BITS, 2>(Xp, P4, Sc, Yp, part, xs[warp], red, e, base + r0, rows, kbase, klen, d, f, ncw, cw, col_ok, warp, lane);
        else               gemv_pass<BITS, 1>(Xp, P4, Sc, Yp, part, xs[warp], red, e, base + r0, rows, kbase, klen, d, f, ncw, cw, col_ok, warp, lane);
    }
}

// Sum the K-split partials in a fixed order (deterministic) and convert to fp16.
__global__ void moe_splitk_reduce(const float* ws, half* Yp, size_t n, int ksplit) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float s = 0.f;
        for (int z = 0; z < ksplit; z++) s += ws[(size_t)z * n + i];
        Yp[i] = __float2half(s);
    }
}

__global__ void moe_scan_offsets(const int* counts, int* offsets, int* cursor, int E) {
    if (threadIdx.x != 0) return;
    int o = 0;
    for (int e = 0; e < E; e++) { offsets[e] = o; o += counts[e]; cursor[e] = 0; }
    offsets[E] = o;
}

// ------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------
struct QuantWeights {
    int bits;
    uint32_t* P = nullptr;
    half* Sc = nullptr;
    size_t words = 0, scales = 0;
    QuantWeights(int bits_, const half* W, int E, int d, int f) : bits(bits_) {
        int cpt = 32 / bits;
        words = (size_t)E * d * (f / cpt);
        CUDA_CHECK(cudaMalloc(&P, words * 4));
        if (bits < 16) {
            scales = (size_t)E * (d / GS) * f;
            CUDA_CHECK(cudaMalloc(&Sc, scales * 2));
        }
        if (bits == 16) pack_weights<16><<<1024, 256>>>(W, P, Sc, E, d, f);
        if (bits == 8)  pack_weights<8> <<<1024, 256>>>(W, P, Sc, E, d, f);
        if (bits == 4)  pack_weights<4> <<<1024, 256>>>(W, P, Sc, E, d, f);
        CUDA_CHECK_KERNEL();
    }
    ~QuantWeights() { cudaFree(P); if (Sc) cudaFree(Sc); if (ws) cudaFree(ws); }
    size_t bytes_per_expert(int d, int f) const {          // what the kernel must stream
        return (size_t)d * f * bits / 8 + (bits < 16 ? (size_t)(d / GS) * f * 2 : 0);
    }
    // Split K across blocks when too few (expert, column-block) pairs would be busy to keep
    // enough loads in flight. Uses only host-known sizes (A = T*k), so no host sync.
    // force_ksplit > 0 overrides the heuristic.
    int pick_ksplit(int A, int E, int d, int f, int force_ksplit) const {
        if (force_ksplit > 0) return std::min(force_ksplit, d / (NW * GS));   // each warp needs >= 1 group
        int dev, sms;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        int busy = ceil_div(f / (32 / bits), 32) * std::min(E, A);
        int ks = 1;
        while (ks * 2 <= d / (NW * GS) && busy * ks < 8 * sms) ks *= 2;
        return ks;
    }
    mutable float* ws = nullptr;
    mutable size_t ws_cap = 0;
    void run(const half* Xp, half* Yp, const int* offsets, int A, int E, int d, int f,
             int force_ksplit = 0, cudaStream_t s = 0) const {
        int ncw = f / (32 / bits);
        int ks = pick_ksplit(A, E, d, f, force_ksplit);
        if (ks > 1 && (size_t)ks * A * f > ws_cap) {
            if (ws) CUDA_CHECK(cudaFree(ws));
            ws_cap = (size_t)ks * A * f;
            CUDA_CHECK(cudaMalloc(&ws, ws_cap * sizeof(float)));
        }
        dim3 grid(ceil_div(ncw, 32), E, ks);
        if (bits == 16) moe_wq_gemv<16><<<grid, NTW, 0, s>>>(Xp, P, Sc, Yp, offsets, d, f, ks, ws, A);
        if (bits == 8)  moe_wq_gemv<8> <<<grid, NTW, 0, s>>>(Xp, P, Sc, Yp, offsets, d, f, ks, ws, A);
        if (bits == 4)  moe_wq_gemv<4> <<<grid, NTW, 0, s>>>(Xp, P, Sc, Yp, offsets, d, f, ks, ws, A);
        if (ks > 1) moe_splitk_reduce<<<256, 256, 0, s>>>(ws, Yp, (size_t)A * f, ks);
    }
};

// Host-side copy of the dequantized weight the kernel uses for (e, k, col).
float host_weight(int bits, const std::vector<uint32_t>& P, const std::vector<half>& Sc,
                  int e, int k, int col, int d, int f) {
    int cpt = 32 / bits, ncw = f / cpt, cw = col / cpt, c = col % cpt;
    uint32_t w = P[widx(e, k, cw, d, ncw)];
    if (bits == 16) {
        __half_raw r;
        r.x = (unsigned short)(w >> (16 * c));
        return __half2float(half(r));
    }
    float s = __half2float(Sc[((size_t)e * (d / GS) + k / GS) * f + col]);
    if (bits == 8) return (float)((int)((w >> (8 * c)) & 0xFF) - 128) * s;
    return (float)((int)((w >> (4 * c)) & 0xF) - 8) * s;
}

struct MoeDecode {
    int T, E, k, d, f;
    half *X, *W, *Xp, *Yp, *out;
    float *logits, *wts;
    int *ids, *counts, *offsets, *cursor, *inv_perm;
    MoeDecode(int T_, int E_, int k_, int d_, int f_) : T(T_), E(E_), k(k_), d(d_), f(f_) {
        if (E > MAX_E || k > MAX_K || d % (NW * GS) || f % 8) { fprintf(stderr, "unsupported shape\n"); exit(1); }
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
    }
    ~MoeDecode() {
        cudaFree(X); cudaFree(W); cudaFree(Xp); cudaFree(Yp); cudaFree(out); cudaFree(logits);
        cudaFree(wts); cudaFree(ids); cudaFree(inv_perm); cudaFree(counts); cudaFree(cursor); cudaFree(offsets);
    }
    void init(unsigned seed, int disable = -1) {
        auto lg = make_logits(T, E, seed, disable);
        CUDA_CHECK(cudaMemcpy(logits, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
        init_uniform<<<1024, 256>>>(X, (size_t)T * d, seed + 1, 1.0f);
        init_uniform<<<1024, 256>>>(W, (size_t)E * d * f, seed + 2, 1.0f / std::sqrt((float)d));
        CUDA_CHECK_KERNEL();
    }
    void route_and_permute(cudaStream_t s = 0) {
        int A = T * k;
        router_topk<<<ceil_div(T * 32, 256), 256, 0, s>>>(logits, ids, wts, T, E, k);
        CUDA_CHECK(cudaMemsetAsync(counts, 0, E * 4, s));
        moe_count<<<ceil_div(A, 256), 256, 0, s>>>(ids, counts, A);
        moe_scan_offsets<<<1, 32, 0, s>>>(counts, offsets, cursor, E);
        moe_permute<<<A, 128, 0, s>>>(X, ids, offsets, cursor, inv_perm, Xp, k, d);
    }
    void combine(cudaStream_t s = 0) { moe_combine<<<T, 256, 0, s>>>(Yp, inv_perm, wts, out, k, f); }
};

// cuBLAS grouped GEMM over the experts that received tokens (fp16 in, fp32 accumulate).
struct CublasGrouped {
    std::vector<cublasOperation_t> ta, tb;
    std::vector<int> gm, gn, gk, glda, gldb, gldc, gsize;
    std::vector<float> ga, gb;
    void **dA = nullptr, **dB = nullptr, **dC = nullptr;
    int G = 0;
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    CublasGrouped(const MoeDecode& m, const std::vector<int>& counts, const std::vector<int>& offs, half* Yout) {
        std::vector<const void*> pA, pB;
        std::vector<void*> pC;
        for (int e = 0; e < m.E; e++) {
            if (!counts[e]) continue;
            ta.push_back(CUBLAS_OP_N); tb.push_back(CUBLAS_OP_N);
            gm.push_back(m.f); gn.push_back(counts[e]); gk.push_back(m.d);
            glda.push_back(m.f); gldb.push_back(m.d); gldc.push_back(m.f);
            ga.push_back(1.f); gb.push_back(0.f); gsize.push_back(1);
            pA.push_back(m.W + (size_t)e * m.d * m.f);
            pB.push_back(m.Xp + (size_t)offs[e] * m.d);
            pC.push_back(Yout + (size_t)offs[e] * m.f);
        }
        G = (int)gsize.size();
        CUDA_CHECK(cudaMalloc(&dA, G * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&dB, G * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&dC, G * sizeof(void*)));
        CUDA_CHECK(cudaMemcpy(dA, pA.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, pB.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, pC.data(), G * sizeof(void*), cudaMemcpyHostToDevice));
    }
    ~CublasGrouped() { cudaFree(dA); cudaFree(dB); cudaFree(dC); }
    void run(cublasHandle_t h) {
        st = cublasGemmGroupedBatchedEx(h, ta.data(), tb.data(), gm.data(), gn.data(), gk.data(),
                                        ga.data(), (const void* const*)dA, CUDA_R_16F, glda.data(),
                                        (const void* const*)dB, CUDA_R_16F, gldb.data(),
                                        gb.data(), (void* const*)dC, CUDA_R_16F, gldc.data(),
                                        G, gsize.data(), CUBLAS_COMPUTE_32F);
    }
};

double sqnr_db(const std::vector<float>& got, const std::vector<float>& ref) {
    double sig = 0, noise = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double e = (double)got[i] - ref[i];
        sig += (double)ref[i] * ref[i];
        noise += e * e;
    }
    return 10.0 * std::log10(sig / std::max(noise, 1e-30));
}

int main() {
    Bench bench;
    cublasHandle_t h;
    cublasCreate(&h);

    // ================= correctness: kernel vs CPU on the SAME dequantized weights =================
    printf("\n[correctness: kernel output vs fp64 CPU GEMM over the identical (de)quantized weights]\n");
    {
        int T = 40, E = 8, K = 2, d = 2048, f = 200;              // f=200: tail column block; d allows 4-way K split
        MoeDecode m(T, E, K, d, f);
        m.init(5, /*disable=*/3);                                 // one empty expert
        m.route_and_permute();
        CUDA_CHECK_KERNEL();
        int A = T * K;
        std::vector<int> offs(E + 1);
        CUDA_CHECK(cudaMemcpy(offs.data(), m.offsets, (E + 1) * 4, cudaMemcpyDeviceToHost));
        std::vector<float> xp = to_float(m.Xp, (size_t)A * d);
        int maxn = 0;
        for (int e = 0; e < E; e++) maxn = std::max(maxn, offs[e + 1] - offs[e]);
        for (int bits : {16, 8, 4})
        for (int ks : {1, 4}) {
            QuantWeights q(bits, m.W, E, d, f);
            CUDA_CHECK(cudaMemset(m.Yp, 0, (size_t)A * f * 2));
            q.run(m.Xp, m.Yp, m.offsets, A, E, d, f, ks);
            CUDA_CHECK_KERNEL();
            std::vector<uint32_t> P(q.words);
            std::vector<half> Sc(q.scales);
            CUDA_CHECK(cudaMemcpy(P.data(), q.P, q.words * 4, cudaMemcpyDeviceToHost));
            if (q.scales) CUDA_CHECK(cudaMemcpy(Sc.data(), q.Sc, q.scales * 2, cudaMemcpyDeviceToHost));
            std::vector<float> ref((size_t)A * f, 0.f);
            for (int e = 0; e < E; e++)
                for (int row = offs[e]; row < offs[e + 1]; row++)
                    for (int col = 0; col < f; col++) {
                        double acc = 0;
                        for (int kk = 0; kk < d; kk++)
                            acc += (double)xp[(size_t)row * d + kk] * host_weight(bits, P, Sc, e, kk, col, d, f);
                        ref[(size_t)row * f + col] = (float)acc;
                    }
            std::vector<float> got = to_float(m.Yp, (size_t)A * f);
            char tag[96];
            snprintf(tag, sizeof tag, "W%dA16 kernel, K split %d ways (rows/expert up to %d, one expert empty)", bits, ks, maxn);
            bench.verify(got.data(), ref.data(), got.size(), tag, 1e-2f, 2e-3f);
        }
    }

    // ================= performance: decode shapes =================
    struct Cfg { const char* name; int T, E, k, d, f; };
    std::vector<Cfg> cfgs = {
        {"fine-grained MoE  E=64 k=6 d=2048 f=1408   T=1 ", 1, 64, 6, 2048, 1408},
        {"fine-grained MoE  E=64 k=6 d=2048 f=1408   T=16", 16, 64, 6, 2048, 1408},
        {"fine-grained MoE  E=64 k=6 d=2048 f=1408   T=64", 64, 64, 6, 2048, 1408},
        {"Mixtral-like      E=8  k=2 d=4096 f=14336  T=1 ", 1, 8, 2, 4096, 14336},
        {"Mixtral-like      E=8  k=2 d=4096 f=14336  T=16", 16, 8, 2, 4096, 14336},
    };
    bool first_cfg = true;
    for (auto& c : cfgs) {
        printf("\n=====================================================================\n");
        printf(" %s\n", c.name);
        MoeDecode m(c.T, c.E, c.k, c.d, c.f);
        m.init(9);
        m.route_and_permute();
        CUDA_CHECK_KERNEL();
        int A = c.T * c.k;
        std::vector<int> counts(c.E), offs(c.E + 1);
        CUDA_CHECK(cudaMemcpy(counts.data(), m.counts, c.E * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(offs.data(), m.offsets, (c.E + 1) * 4, cudaMemcpyDeviceToHost));
        int active = 0, maxn = 0;
        for (int n : counts) { active += (n > 0); maxn = std::max(maxn, n); }
        printf(" %d rows over %d active experts (max %d rows on one expert)\n", A, active, maxn);
        // same per-expert row counts for the PyTorch baselines (tools/torch_baselines.py)
        if (FILE* fc = fopen("results/moe_decode_counts.txt", first_cfg ? "w" : "a")) {
            fprintf(fc, "%d %d %d %d", c.T, c.E, c.d, c.f);
            for (int n : counts) fprintf(fc, " %d", n);
            fprintf(fc, "\n");
            fclose(fc);
        }
        first_cfg = false;

        half* Yref;
        CUDA_CHECK(cudaMalloc(&Yref, (size_t)A * c.f * 2));
        CublasGrouped cg(m, counts, offs, Yref);
        cg.run(h);
        CUDA_CHECK_KERNEL();
        if (cg.st != CUBLAS_STATUS_SUCCESS) { printf(" cuBLAS grouped GEMM failed (status %d)\n", (int)cg.st); return 1; }
        std::vector<float> ref = to_float(Yref, (size_t)A * c.f);
        double fp16_bytes = (double)active * c.d * c.f * 2;

        // all three weight formats resident at once, so they can be timed interleaved with cuBLAS
        QuantWeights q16(16, m.W, c.E, c.d, c.f), q8(8, m.W, c.E, c.d, c.f), q4(4, m.W, c.E, c.d, c.f);
        const QuantWeights* qs[3] = {&q16, &q8, &q4};
        double sqnr[3];
        for (int i = 0; i < 3; i++) {
            qs[i]->run(m.Xp, m.Yp, m.offsets, A, c.E, c.d, c.f);
            CUDA_CHECK_KERNEL();
            sqnr[i] = sqnr_db(to_float(m.Yp, (size_t)A * c.f), ref);
        }
        std::vector<float> tm = bench.time_interleaved(
            {[&] { cg.run(h); },
             [&] { q16.run(m.Xp, m.Yp, m.offsets, A, c.E, c.d, c.f); },
             [&] { q8.run(m.Xp, m.Yp, m.offsets, A, c.E, c.d, c.f); },
             [&] { q4.run(m.Xp, m.Yp, m.offsets, A, c.E, c.d, c.f); }},
            10, 3, 3, /*cold=*/true);
        float t_cub = tm[0];

        printf("   %-34s %9s %10s %12s %10s\n", "expert GEMM stage", "time(ms)", "weight MB", "weight GB/s", "vs cuBLAS");
        printf("   %-34s %9.4f %10.1f %12.1f %9.2fx\n", "cuBLAS grouped GEMM, fp16 weights", t_cub,
               fp16_bytes / 1e6, fp16_bytes / t_cub / 1e6, 1.0);
        const char* names[3] = {"fp16 weights", "W8A16 (int8, g128)", "W4A16 (int4, g128)"};
        for (int i = 0; i < 3; i++) {
            double bytes = (double)active * qs[i]->bytes_per_expert(c.d, c.f);
            char nm[64];
            snprintf(nm, sizeof nm, "this repo, %s ks=%d", names[i], qs[i]->pick_ksplit(A, c.E, c.d, c.f, 0));
            printf("   %-34s %9.4f %10.1f %12.1f %9.2fx   SQNR vs fp16: %.1f dB\n", nm, tm[i + 1], bytes / 1e6,
                   bytes / tm[i + 1] / 1e6, t_cub / tm[i + 1], sqnr[i]);
        }
        float t_full = bench.time_cold([&] {
            m.route_and_permute();
            q4.run(m.Xp, m.Yp, m.offsets, A, c.E, c.d, c.f);
            m.combine();
        }, 5, 30);
        printf("   full layer with W4A16 (route + permute + GEMM + combine, no host sync): %.4f ms\n", t_full);
        CUDA_CHECK(cudaFree(Yref));
    }

    // ================= end to end: what does "no host sync" buy? =================
    // A library-built MoE layer has to bring the per-expert row counts to the host after
    // routing, because cuBLAS grouped GEMM takes its problem sizes from host arrays. That
    // device->host copy stalls the stream every layer, and the data-dependent host logic means
    // the layer cannot be captured in a CUDA Graph. This repo's layer never leaves the GPU.
    //   cuBLAS + host sync : route/permute -> D2H counts + sync -> host builds args -> cuBLAS grouped -> combine
    //   ours, eager        : route/permute -> grouped GEMV -> combine, launched one by one
    //   ours, CUDA Graph   : the same layer(s) captured once and replayed
    // L layers each get their own weights (as in a real model); L2 is flushed before every sample.
    printf("\n=====================================================================\n");
    printf(" end to end: host-synced cuBLAS layer vs device-scheduled layer (+ CUDA Graph)\n");
    printf(" fine-grained MoE E=64 k=6 d=2048 f=1408, fp16 activations, interleaved cold-L2 timing\n");
    {
        const int E = 64, K = 6, d = 2048, f = 1408, LMAX = 4;
        cudaStream_t s;
        CUDA_CHECK(cudaStreamCreate(&s));                   // blocking stream: orders with the timing events
        cublasSetStream(h, s);
        int* h_counts;
        void** h_ptr;
        void** d_ptr;
        CUDA_CHECK(cudaMallocHost(&h_counts, E * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_ptr, 3 * E * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&d_ptr, 3 * E * sizeof(void*)));
        std::vector<cublasOperation_t> ta(E, CUBLAS_OP_N), tb(E, CUBLAS_OP_N);
        std::vector<int> gm(E, f), gn(E), gk(E, d), glda(E, f), gldb(E, d), gldc(E, f), gsize(E, 1);
        std::vector<float> ga(E, 1.f), gb(E, 0.f);

        for (int T : {1, 16}) {
            MoeDecode m(T, E, K, d, f);
            m.init(9);
            const int A = T * K;
            std::vector<half*> W(LMAX);
            W[0] = m.W;
            for (int l = 1; l < LMAX; l++) {
                CUDA_CHECK(cudaMalloc(&W[l], (size_t)E * d * f * 2));
                init_uniform<<<1024, 256>>>(W[l], (size_t)E * d * f, 100 + l, 1.0f / std::sqrt((float)d));
            }
            CUDA_CHECK_KERNEL();
            std::vector<QuantWeights*> q16(LMAX), q4(LMAX);
            for (int l = 0; l < LMAX; l++) {
                q16[l] = new QuantWeights(16, W[l], E, d, f);
                q4[l]  = new QuantWeights(4,  W[l], E, d, f);
            }

            auto cublas_layer = [&](int l) {
                m.route_and_permute(s);
                CUDA_CHECK(cudaMemcpyAsync(h_counts, m.counts, E * sizeof(int), cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaStreamSynchronize(s));       // <- the stall this experiment measures
                int G = 0, off = 0;
                for (int e = 0; e < E; e++) {
                    if (h_counts[e]) {
                        gn[G] = h_counts[e];
                        h_ptr[G]         = W[l] + (size_t)e * d * f;
                        h_ptr[E + G]     = m.Xp + (size_t)off * d;
                        h_ptr[2 * E + G] = m.Yp + (size_t)off * f;
                        G++;
                    }
                    off += h_counts[e];
                }
                CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, 3 * E * sizeof(void*), cudaMemcpyHostToDevice, s));
                cublasGemmGroupedBatchedEx(h, ta.data(), tb.data(), gm.data(), gn.data(), gk.data(), ga.data(),
                                           (const void* const*)d_ptr, CUDA_R_16F, glda.data(),
                                           (const void* const*)(d_ptr + E), CUDA_R_16F, gldb.data(), gb.data(),
                                           (void* const*)(d_ptr + 2 * E), CUDA_R_16F, gldc.data(),
                                           G, gsize.data(), CUBLAS_COMPUTE_32F);
                m.combine(s);
            };
            auto our_layer = [&](const QuantWeights* q) {
                m.route_and_permute(s);
                q->run(m.Xp, m.Yp, m.offsets, A, E, d, f, 0, s);
                m.combine(s);
            };

            // correctness: the three paths must agree on the layer output
            std::vector<float> o_cub, o_our;
            cublas_layer(0);
            CUDA_CHECK(cudaStreamSynchronize(s));
            o_cub = to_float(m.out, (size_t)T * f);
            our_layer(q16[0]);
            CUDA_CHECK(cudaStreamSynchronize(s));
            o_our = to_float(m.out, (size_t)T * f);
            char tag[96];
            snprintf(tag, sizeof tag, "T=%-2d layer output: ours (fp16) vs host-synced cuBLAS", T);
            bench.verify(o_our.data(), o_cub.data(), o_our.size(), tag, 1e-2f, 2e-3f);

            for (int L : {1, LMAX}) {
                // capture L layers once; every workspace was already allocated by the eager runs above
                for (int l = 0; l < L; l++) { our_layer(q16[l]); our_layer(q4[l]); }
                CUDA_CHECK(cudaStreamSynchronize(s));
                cudaGraph_t g16, g4;
                cudaGraphExec_t x16, x4;
                CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
                for (int l = 0; l < L; l++) our_layer(q16[l]);
                CUDA_CHECK(cudaStreamEndCapture(s, &g16));
                CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
                for (int l = 0; l < L; l++) our_layer(q4[l]);
                CUDA_CHECK(cudaStreamEndCapture(s, &g4));
                CUDA_CHECK(cudaGraphInstantiate(&x16, g16, 0));
                CUDA_CHECK(cudaGraphInstantiate(&x4, g4, 0));

                std::vector<float> tm = bench.time_interleaved(
                    {[&] { for (int l = 0; l < L; l++) cublas_layer(l); },
                     [&] { for (int l = 0; l < L; l++) our_layer(q16[l]); },
                     [&] { CUDA_CHECK(cudaGraphLaunch(x16, s)); },
                     [&] { CUDA_CHECK(cudaGraphLaunch(x4, s)); }},
                    10, 3, 3, /*cold=*/true);
                printf("\n T=%d, %d layer%s (time per layer)\n", T, L, L > 1 ? "s" : "");
                const char* nm[4] = {"cuBLAS grouped FP16, host sync per layer",
                                     "ours FP16, eager launches, no sync     ",
                                     "ours FP16, CUDA Graph                  ",
                                     "ours W4A16, CUDA Graph                 "};
                for (int i = 0; i < 4; i++)
                    printf("   %s  %8.4f ms   %5.2fx\n", nm[i], tm[i] / L, tm[0] / tm[i]);
                CUDA_CHECK(cudaGraphExecDestroy(x16)); CUDA_CHECK(cudaGraphExecDestroy(x4));
                CUDA_CHECK(cudaGraphDestroy(g16));     CUDA_CHECK(cudaGraphDestroy(g4));
            }
            for (int l = 0; l < LMAX; l++) { delete q16[l]; delete q4[l]; }
            for (int l = 1; l < LMAX; l++) CUDA_CHECK(cudaFree(W[l]));
        }
        cublasSetStream(h, 0);
        CUDA_CHECK(cudaFreeHost(h_counts)); CUDA_CHECK(cudaFreeHost(h_ptr)); CUDA_CHECK(cudaFree(d_ptr));
        CUDA_CHECK(cudaStreamDestroy(s));
    }

    cublasDestroy(h);
    return bench.finish();
}

// moe.cuh -- MoE routing / permute / combine pieces shared by the MoE experiments
#pragma once
#include "bench.cuh"
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <random>

constexpr int MAX_E = 256;
constexpr int MAX_K = 8;
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- routing
// One warp per token. Lane l holds experts l, l+32, ... (E <= 256).
__global__ void router_topk(const float* logits, int* ids, float* wts, int T, int E, int k) {
    const int tok  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (tok >= T) return;                                   // uniform per warp
    const float* lg = logits + (size_t)tok * E;

    float v[MAX_E / 32];
    float mx = -INFINITY;
    #pragma unroll
    for (int i = 0; i < MAX_E / 32; i++) {
        int e = lane + 32 * i;
        v[i] = (e < E) ? lg[e] : -INFINITY;
        mx = fmaxf(mx, v[i]);
    }
    mx = warp_max(mx);
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < MAX_E / 32; i++) {
        int e = lane + 32 * i;
        v[i] = (e < E) ? __expf(v[i] - mx) : -1.f;          // probs >= 0; -1 marks "unavailable"
        s += fmaxf(v[i], 0.f);
    }
    s = warp_sum(s);

    float picked = 0.f, my_p[MAX_K];
    int my_id[MAX_K];
    for (int r = 0; r < k; r++) {
        float best = -1.f;
        int bi = INT_MAX;
        #pragma unroll
        for (int i = 0; i < MAX_E / 32; i++)
            if (v[i] > best) { best = v[i]; bi = lane + 32 * i; }
        for (int o = 16; o > 0; o >>= 1) {                  // warp argmax, ties -> lower id
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            int   oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        #pragma unroll
        for (int i = 0; i < MAX_E / 32; i++)
            if (lane + 32 * i == bi) v[i] = -1.f;           // owner removes the winner
        my_id[r] = bi;
        my_p[r]  = best / s;
        picked  += best / s;
    }
    if (lane == 0)
        for (int r = 0; r < k; r++) {
            ids[tok * k + r] = my_id[r];
            wts[tok * k + r] = my_p[r] / picked;            // renormalize over the selected k
        }
}

__global__ void moe_count(const int* ids, int* counts, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[ids[i]], 1);
}

// One block per (token, slot) assignment.
__global__ void moe_permute(const half* X, const int* ids, const int* offsets, int* cursor,
                            int* inv_perm, half* Xp, int k, int d) {
    __shared__ int pos_s;
    const int i = blockIdx.x;
    if (threadIdx.x == 0) {
        int e = ids[i];
        pos_s = offsets[e] + atomicAdd(&cursor[e], 1);      // integer slot allocation
        inv_perm[i] = pos_s;
    }
    __syncthreads();
    const uint4* src = reinterpret_cast<const uint4*>(X + (size_t)(i / k) * d);
    uint4*       dst = reinterpret_cast<uint4*>(Xp + (size_t)pos_s * d);
    for (int c = threadIdx.x; c < d / 8; c += blockDim.x) dst[c] = src[c];
}

// ---------------------------------------------------------------- combine
// Gather: each token reads its k expert outputs in slot order -> deterministic.
__global__ void moe_combine(const half* Yp, const int* inv_perm, const float* wts,
                            half* out, int k, int f) {
    const int tok = blockIdx.x;
    for (int c = threadIdx.x; c < f / 8; c += blockDim.x) {
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int j = 0; j < k; j++) {
            int pos = inv_perm[tok * k + j];
            float w = wts[tok * k + j];
            uint4 raw = reinterpret_cast<const uint4*>(Yp + (size_t)pos * f)[c];
            const half2* h = reinterpret_cast<const half2*>(&raw);
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                float2 v = __half22float2(h[q]);
                acc[2 * q] += w * v.x;
                acc[2 * q + 1] += w * v.y;
            }
        }
        __align__(16) half2 o[4];
        #pragma unroll
        for (int q = 0; q < 4; q++) o[q] = __floats2half2_rn(acc[2 * q], acc[2 * q + 1]);
        reinterpret_cast<uint4*>(out + (size_t)tok * f)[c] = *reinterpret_cast<uint4*>(o);
    }
}

// Scatter: each assignment atomically adds into its token row -> order (and bits) vary.
__global__ void moe_combine_atomic(const half* Yp, const int* inv_perm, const float* wts,
                                   float* out, int k, int f) {
    const int i = blockIdx.x, tok = i / k;
    const int pos = inv_perm[i];
    const float w = wts[i];
    for (int c = threadIdx.x; c < f; c += blockDim.x)
        atomicAdd(&out[(size_t)tok * f + c], w * __half2float(Yp[(size_t)pos * f + c]));
}

__global__ void init_uniform(half* p, size_t n, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)(i ^ (i >> 32)) * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        float u = (x >> 8) * (1.0f / 16777216.0f);
        p[i] = __float2half((2.f * u - 1.f) * scale);
    }
}

// Random logits with a per-expert bias so routing is skewed, like real traces.
// disable >= 0 forces that expert to receive no tokens (tests the empty-expert path).
inline std::vector<float> make_logits(int T, int E, unsigned seed, int disable = -1) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> N01(0.f, 1.f);
    std::vector<float> bias(E), lg((size_t)T * E);
    for (auto& b : bias) b = 0.8f * N01(rng);
    if (disable >= 0) bias[disable] = -1e4f;
    for (int t = 0; t < T; t++)
        for (int e = 0; e < E; e++) lg[(size_t)t * E + e] = N01(rng) + bias[e];
    return lg;
}

inline std::vector<float> to_float(const half* dptr, size_t n) {
    std::vector<half> h(n);
    std::vector<float> f(n);
    CUDA_CHECK(cudaMemcpy(h.data(), dptr, n * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) f[i] = __half2float(h[i]);
    return f;
}

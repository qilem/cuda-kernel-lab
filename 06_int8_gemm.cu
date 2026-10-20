// 06_int8_gemm.cu
//
// W8A8 quantized GEMM, written out explicitly:
//   weights     per-output-channel SYMMETRIC int8:   s = max|w| / 127,  z = 0
//   activations per-token ASYMMETRIC uint8:          s = (max - min) / 255, z = round(-min / s)
//
// Integer GEMM with zero points, expanded so only one term lives in the K loop:
//   sum_k (qa - za)(qw - zw) = sum qa*qw  -  zw * sum qa  -  za * sum qw  +  K * za * zw
//                              main term     row sums        col sums       constant
// The row/column sums are produced by the quantization kernels (column sums of fixed
// weights would be computed offline in a real deployment).
//
// Why scales are grouped along M and N but never along K: a scale can be pulled out of
// sum_k only if it is constant along k. Per-token (m) and per-channel (n) scales are; a
// per-k scale is not, and would force a dequantize inside the K loop.
//
// Validation is two-level: (A) the expanded integer formula must match the definition
// bit-exactly, (B) the full float -> int8 -> GEMM -> float pipeline is judged by SQNR
// against an fp64 GEMM, because element-wise allclose is meaningless after quantization.

#include "common/bench.cuh"
#include <cstdint>

__device__ float warp_max(float v) {
    for (int off = 16; off > 0; off /= 2) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ float block_max(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    v = warp_max(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < nw) ? s[threadIdx.x] : -INFINITY;
        v = warp_max(v);
        if (threadIdx.x == 0) s[0] = v;
    }
    __syncthreads();
    float r = s[0];
    __syncthreads();                                   // called twice in one kernel
    return r;
}
__device__ float block_sum(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    v = warp_sum(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < nw) ? s[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) s[0] = v;
    }
    __syncthreads();
    float r = s[0];
    __syncthreads();
    return r;
}

// W is [N, K]; one block per output channel n.
__global__ void quant_weight_sym(const float* w, int8_t* qw, float* scale, int* qsum, int K) {
    int n = blockIdx.x;
    const float* wr = w + (size_t)n * K;
    int8_t* qr = qw + (size_t)n * K;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf(wr[k]));
    m = block_max(m);
    float s = (m > 0.0f) ? m / 127.0f : 1.0f;           // all-zero row: avoid divide by zero
    float local = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int q = __float2int_rn(wr[k] / s);
        q = max(-127, min(127, q));                     // symmetric range: -128 unused
        qr[k] = (int8_t)q;
        local += q;
    }
    float tot = block_sum(local);                       // integer sum, exact while < 2^24
    if (threadIdx.x == 0) { scale[n] = s; qsum[n] = (int)tot; }
}

// X is [M, K]; one block per token m.
__global__ void quant_act_asym(const float* x, uint8_t* qx, float* scale, int* zp, int* qsum, int K) {
    int m = blockIdx.x;
    const float* xr = x + (size_t)m * K;
    uint8_t* qr = qx + (size_t)m * K;
    float mx = -INFINITY, mn = INFINITY;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        mx = fmaxf(mx, xr[k]);
        mn = fminf(mn, xr[k]);
    }
    mx = block_max(mx);
    mn = -block_max(-mn);                               // min(x) = -max(-x)
    mx = fmaxf(mx, 0.0f);                               // range must contain 0 so that 0 is exact
    mn = fminf(mn, 0.0f);
    float s = (mx > mn) ? (mx - mn) / 255.0f : 1.0f;
    int z = __float2int_rn(-mn / s);
    float local = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int q = __float2int_rn(xr[k] / s) + z;
        q = max(0, min(255, q));
        qr[k] = (uint8_t)q;
        local += q;
    }
    float tot = block_sum(local);
    if (threadIdx.x == 0) { scale[m] = s; zp[m] = z; qsum[m] = (int)tot; }
}

// Y[m][n] = sa[m] * sw[n] * sum_k (qa - za)(qw - zw), via the four-term expansion.
__global__ void gemm_int8(const uint8_t* qa, const int8_t* qw,
                          const int* za, const int* zw, const int* rowsum, const int* colsum,
                          const float* sa, const float* sw,
                          int* acc_out, float* y, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        int dot = 0;
        for (int k = 0; k < K; k++)
            dot += (int)qa[(size_t)m * K + k] * (int)qw[(size_t)n * K + k];
        int acc = dot - zw[n] * rowsum[m] - za[m] * colsum[n] + K * za[m] * zw[n];
        acc_out[(size_t)m * N + n] = acc;
        y[(size_t)m * N + n] = (float)acc * sa[m] * sw[n];
    }
}

int main() {
    Bench b;
    const int T = 256;

    {   // (A) expansion is exact in int32
        printf("\n[A: four-term expansion vs direct definition, exact int32 compare]\n");
        int M = 64, N = 64, K = 1000;
        std::vector<uint8_t> qa((size_t)M * K);
        std::vector<int8_t> qw((size_t)N * K);
        std::vector<int> za(M), zw(N), rowsum(M, 0), colsum(N, 0);
        srand(101);
        for (auto& v : qa) v = (uint8_t)(rand() % 256);
        for (auto& v : qw) v = (int8_t)(rand() % 255 - 127);
        for (auto& v : za) v = rand() % 256;
        for (auto& v : zw) v = rand() % 41 - 20;       // nonzero weight zero points: all 4 terms active
        for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) rowsum[m] += qa[(size_t)m * K + k];
        for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) colsum[n] += qw[(size_t)n * K + k];
        std::vector<int> ref((size_t)M * N);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                long long s = 0;
                for (int k = 0; k < K; k++)
                    s += (long long)(qa[(size_t)m * K + k] - za[m]) * (qw[(size_t)n * K + k] - zw[n]);
                ref[(size_t)m * N + n] = (int)s;
            }
        uint8_t* dqa; int8_t* dqw; int *dza, *dzw, *drs, *dcs, *dacc; float *dsa, *dsw, *dy;
        CUDA_CHECK(cudaMalloc(&dqa, qa.size())); CUDA_CHECK(cudaMalloc(&dqw, qw.size()));
        CUDA_CHECK(cudaMalloc(&dza, M * 4)); CUDA_CHECK(cudaMalloc(&dzw, N * 4));
        CUDA_CHECK(cudaMalloc(&drs, M * 4)); CUDA_CHECK(cudaMalloc(&dcs, N * 4));
        CUDA_CHECK(cudaMalloc(&dacc, (size_t)M * N * 4)); CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 4));
        CUDA_CHECK(cudaMalloc(&dsa, M * 4)); CUDA_CHECK(cudaMalloc(&dsw, N * 4));
        CUDA_CHECK(cudaMemcpy(dqa, qa.data(), qa.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dqw, qw.data(), qw.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dza, za.data(), M * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dzw, zw.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(drs, rowsum.data(), M * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dcs, colsum.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dsa, 0, M * 4)); CUDA_CHECK(cudaMemset(dsw, 0, N * 4));
        dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
        gemm_int8<<<grd, blk>>>(dqa, dqw, dza, dzw, drs, dcs, dsa, dsw, dacc, dy, M, N, K);
        CUDA_CHECK_KERNEL();
        std::vector<int> got((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(got.data(), dacc, got.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (size_t t = 0; t < got.size(); t++) bad += (got[t] != ref[t]);
        printf("  int32 results differing from the definition: %d / %zu  %s\n", bad, got.size(), bad ? "FAIL" : "ok");
        if (bad) b.n_fail++;
        CUDA_CHECK(cudaFree(dqa)); CUDA_CHECK(cudaFree(dqw)); CUDA_CHECK(cudaFree(dza)); CUDA_CHECK(cudaFree(dzw));
        CUDA_CHECK(cudaFree(drs)); CUDA_CHECK(cudaFree(dcs)); CUDA_CHECK(cudaFree(dacc)); CUDA_CHECK(cudaFree(dy));
        CUDA_CHECK(cudaFree(dsa)); CUDA_CHECK(cudaFree(dsw));
    }

    {   // (B) full pipeline vs fp64 GEMM
        printf("\n[B: float -> quantize -> INT8 GEMM -> dequantize, vs fp64 GEMM]\n");
        int M = 256, N = 512, K = 1024;
        std::vector<float> hx((size_t)M * K), hw((size_t)N * K);
        fill_random(hx.data(), hx.size(), -1.0f, 3.0f, 111);     // activations: skewed positive
        fill_random(hw.data(), hw.size(), -0.5f, 0.5f, 112);     // weights: centered at 0

        std::vector<int8_t> rqw(hw.size());                      // CPU quantization reference
        for (int n = 0; n < N; n++) {
            float m = 0;
            for (int k = 0; k < K; k++) m = fmaxf(m, fabsf(hw[(size_t)n * K + k]));
            float s = (m > 0) ? m / 127.0f : 1.0f;
            for (int k = 0; k < K; k++) {
                int q = (int)nearbyintf(hw[(size_t)n * K + k] / s);
                rqw[(size_t)n * K + k] = (int8_t)std::max(-127, std::min(127, q));
            }
        }
        std::vector<uint8_t> rqx(hx.size());
        for (int m = 0; m < M; m++) {
            float mx = 0, mn = 0;
            for (int k = 0; k < K; k++) { mx = fmaxf(mx, hx[(size_t)m * K + k]); mn = fminf(mn, hx[(size_t)m * K + k]); }
            float s = (mx > mn) ? (mx - mn) / 255.0f : 1.0f;
            int z = (int)nearbyintf(-mn / s);
            for (int k = 0; k < K; k++) {
                int q = (int)nearbyintf(hx[(size_t)m * K + k] / s) + z;
                rqx[(size_t)m * K + k] = (uint8_t)std::max(0, std::min(255, q));
            }
        }
        std::vector<double> yref((size_t)M * N);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                double s = 0;
                for (int k = 0; k < K; k++) s += (double)hx[(size_t)m * K + k] * hw[(size_t)n * K + k];
                yref[(size_t)m * N + n] = s;
            }

        float *dx, *dw, *dsa, *dsw, *dy; uint8_t* dqx; int8_t* dqw;
        int *dza, *dzw, *drs, *dcs, *dacc;
        CUDA_CHECK(cudaMalloc(&dx, hx.size() * 4)); CUDA_CHECK(cudaMalloc(&dw, hw.size() * 4));
        CUDA_CHECK(cudaMalloc(&dqx, hx.size())); CUDA_CHECK(cudaMalloc(&dqw, hw.size()));
        CUDA_CHECK(cudaMalloc(&dsa, M * 4)); CUDA_CHECK(cudaMalloc(&dsw, N * 4));
        CUDA_CHECK(cudaMalloc(&dza, M * 4)); CUDA_CHECK(cudaMalloc(&dzw, N * 4));
        CUDA_CHECK(cudaMalloc(&drs, M * 4)); CUDA_CHECK(cudaMalloc(&dcs, N * 4));
        CUDA_CHECK(cudaMalloc(&dacc, (size_t)M * N * 4)); CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 4));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dzw, 0, N * 4));                  // symmetric weights: zw = 0

        quant_weight_sym<<<N, T>>>(dw, dqw, dsw, dcs, K);
        quant_act_asym<<<M, T>>>(dx, dqx, dsa, dza, drs, K);
        dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
        gemm_int8<<<grd, blk>>>(dqx, dqw, dza, dzw, drs, dcs, dsa, dsw, dacc, dy, M, N, K);
        CUDA_CHECK_KERNEL();

        std::vector<int8_t> gqw(hw.size());
        std::vector<uint8_t> gqx(hx.size());
        std::vector<float> gy((size_t)M * N);
        CUDA_CHECK(cudaMemcpy(gqw.data(), dqw, gqw.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gqx.data(), dqx, gqx.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gy.data(), dy, gy.size() * 4, cudaMemcpyDeviceToHost));
        int badw = 0, badx = 0;
        for (size_t t = 0; t < gqw.size(); t++) badw += (gqw[t] != rqw[t]);
        for (size_t t = 0; t < gqx.size(); t++) badx += (gqx[t] != rqx[t]);
        printf("  int8 weights differing from CPU quantizer : %d / %zu\n", badw, gqw.size());
        printf("  uint8 acts differing from CPU quantizer   : %d / %zu\n", badx, gqx.size());
        if (badw || badx) b.n_fail++;
        double sig = 0, noise = 0;
        for (size_t t = 0; t < gy.size(); t++) {
            double e = gy[t] - yref[t];
            sig += yref[t] * yref[t];
            noise += e * e;
        }
        double sqnr = 10.0 * log10(sig / noise);
        printf("  INT8 GEMM vs fp64 GEMM: SQNR = %.1f dB  (8-bit ideal is ~6 dB/bit; >40 dB expected)\n", sqnr);
        if (sqnr < 40) b.n_fail++;
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dqx)); CUDA_CHECK(cudaFree(dqw));
        CUDA_CHECK(cudaFree(dsa)); CUDA_CHECK(cudaFree(dsw)); CUDA_CHECK(cudaFree(dza)); CUDA_CHECK(cudaFree(dzw));
        CUDA_CHECK(cudaFree(drs)); CUDA_CHECK(cudaFree(dcs)); CUDA_CHECK(cudaFree(dacc)); CUDA_CHECK(cudaFree(dy));
    }
    return b.finish();
}

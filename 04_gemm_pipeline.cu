// 04_gemm_pipeline.cu
//
// Question: does double buffering actually help a Tensor Core GEMM?
//
// Load path, buffering, and compute unit are three independent axes. Here the compute
// unit is fixed (WMMA / Tensor Cores) and the other two are swept:
//
//                      LDG + STS (through registers)    cp.async (global -> shared)
//   single buffer             D                                 F
//   double buffer             E                                 G
//
// Finding: double buffering is only worth much when the copy is asynchronous.
// With synchronous LDG/STS the "prefetch" stalls on the load before the MMA loop can
// start, so very little overlap happens.

#include "common/bench.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#ifdef _MSC_VER
#pragma comment(lib, "cublas.lib")
#endif
using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 32
#define LDA (BK + 8)   // 40 halves = 80 B: multiple of 16 B (cp.async) and skews banks
#define LDB (BN + 8)   // 72 halves = 144 B
#define NWARP_N 4
#define NTHREAD 512    // 16 warps, each owns one 16x16 sub-tile of the 64x64 block tile

// ---- load path 1: plain LDG + STS, data passes through registers ----
__device__ __forceinline__ void load_sync(const half* A, const half* B,
                                          half As[BM][LDA], half Bs[BK][LDB],
                                          int rowBase, int colBase, int k0, int N, int K, int tid) {
    for (int i = tid; i < BM * BK; i += NTHREAD) {
        int r = i / BK, c = i % BK;
        As[r][c] = A[(size_t)(rowBase + r) * K + k0 + c];
    }
    for (int i = tid; i < BK * BN; i += NTHREAD) {
        int r = i / BN, c = i % BN;
        Bs[r][c] = B[(size_t)(k0 + r) * N + colBase + c];
    }
}

// ---- load path 2: cp.async, 16-byte copies global -> shared, no register staging ----
__device__ __forceinline__ void load_async(const half* A, const half* B,
                                           half As[BM][LDA], half Bs[BK][LDB],
                                           int rowBase, int colBase, int k0, int N, int K, int tid) {
    const int cpr_a = BK / 8;                                   // 16-byte chunks per A row
    for (int i = tid; i < BM * cpr_a; i += NTHREAD) {
        int r = i / cpr_a, c = (i % cpr_a) * 8;
        __pipeline_memcpy_async(&As[r][c], &A[(size_t)(rowBase + r) * K + k0 + c], 16);
    }
    const int cpr_b = BN / 8;
    for (int i = tid; i < BK * cpr_b; i += NTHREAD) {
        int r = i / cpr_b, c = (i % cpr_b) * 8;
        __pipeline_memcpy_async(&Bs[r][c], &B[(size_t)(k0 + r) * N + colBase + c], 16);
    }
    __pipeline_commit();                                        // one commit group per stage
}

#define COMPUTE_TILE(As_, Bs_)                                             \
    for (int kk = 0; kk < BK; kk += 16) {                                  \
        wmma::load_matrix_sync(a, &As_[wM * 16][kk], LDA);                 \
        wmma::load_matrix_sync(b, &Bs_[kk][wN * 16], LDB);                 \
        wmma::mma_sync(acc, a, b, acc);                                    \
    }

#define PROLOGUE                                                           \
    int tid = threadIdx.x, wid = tid / 32;                                 \
    int wM = wid / NWARP_N, wN = wid % NWARP_N;                            \
    int rowBase = blockIdx.x * BM, colBase = blockIdx.y * BN;              \
    int nt = K / BK;                                                       \
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;   \
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;   \
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;              \
    wmma::fill_fragment(acc, 0.f);

#define EPILOGUE                                                           \
    wmma::store_matrix_sync(C + (size_t)(rowBase + wM * 16) * N + colBase + wN * 16, \
                            acc, N, wmma::mem_row_major);

// D: single buffer + LDG/STS
__global__ void tc_single_sync(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[BM][LDA], Bs[BK][LDB];
    PROLOGUE
    for (int t = 0; t < nt; t++) {
        load_sync(A, B, As, Bs, rowBase, colBase, t * BK, N, K, tid);
        __syncthreads();
        COMPUTE_TILE(As, Bs)
        __syncthreads();
    }
    EPILOGUE
}

// E: double buffer + LDG/STS
__global__ void tc_double_sync(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[2][BM][LDA], Bs[2][BK][LDB];
    PROLOGUE
    load_sync(A, B, As[0], Bs[0], rowBase, colBase, 0, N, K, tid);
    __syncthreads();
    int buf = 0;
    for (int t = 0; t < nt; t++) {
        int nx = 1 - buf;
        if (t + 1 < nt) load_sync(A, B, As[nx], Bs[nx], rowBase, colBase, (t + 1) * BK, N, K, tid);
        COMPUTE_TILE(As[buf], Bs[buf])
        __syncthreads();
        buf = nx;
    }
    EPILOGUE
}

// F: single buffer + cp.async
__global__ void tc_single_async(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[BM][LDA], Bs[BK][LDB];
    PROLOGUE
    for (int t = 0; t < nt; t++) {
        load_async(A, B, As, Bs, rowBase, colBase, t * BK, N, K, tid);
        __pipeline_wait_prior(0);                   // wait for this stage
        __syncthreads();
        COMPUTE_TILE(As, Bs)
        __syncthreads();
    }
    EPILOGUE
}

// G: double buffer + cp.async (all three axes on)
__global__ void tc_double_async(const half* A, const half* B, float* C, int M, int N, int K) {
    __shared__ half As[2][BM][LDA], Bs[2][BK][LDB];
    PROLOGUE
    load_async(A, B, As[0], Bs[0], rowBase, colBase, 0, N, K, tid);
    int buf = 0;
    for (int t = 0; t < nt; t++) {
        int nx = 1 - buf;
        if (t + 1 < nt) {
            load_async(A, B, As[nx], Bs[nx], rowBase, colBase, (t + 1) * BK, N, K, tid);
            __pipeline_wait_prior(1);               // newest stage may still be in flight
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        COMPUTE_TILE(As[buf], Bs[buf])
        __syncthreads();
        buf = nx;
    }
    EPILOGUE
}

// Row-major C = A B is column-major C^T = B^T A^T.
void cublas_hgemm(cublasHandle_t h, const half* A, const half* B, float* C, int M, int N, int K) {
    const float al = 1.f, be = 0.f;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16F, N,
                 A, CUDA_R_16F, K, &be, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

int main() {
    Bench b;
    cublasHandle_t h; cublasCreate(&h);
    const char* nm[5] = {"D  single buffer + LDG/STS ",
                         "E  double buffer + LDG/STS ",
                         "F  single buffer + cp.async",
                         "G  double buffer + cp.async",
                         "   cuBLAS FP16             "};
    auto launch = [&](int w, const half* dA, const half* dB, float* dC, int M, int N, int K) {
        dim3 g(M / BM, N / BN);
        switch (w) {
            case 0: tc_single_sync <<<g, NTHREAD>>>(dA, dB, dC, M, N, K); break;
            case 1: tc_double_sync <<<g, NTHREAD>>>(dA, dB, dC, M, N, K); break;
            case 2: tc_single_async<<<g, NTHREAD>>>(dA, dB, dC, M, N, K); break;
            case 3: tc_double_async<<<g, NTHREAD>>>(dA, dB, dC, M, N, K); break;
            case 4: cublas_hgemm(h, dA, dB, dC, M, N, K); break;
        }
    };

    {   // correctness (inputs rounded to fp16 first, so the only difference is summation order)
        int M = 256, N = 256, K = 256;
        printf("\n[correctness %dx%dx%d]\n", M, N, K);
        std::vector<float> fA((size_t)M * K), fB((size_t)K * N), hC((size_t)M * N), ref((size_t)M * N);
        fill_random(fA.data(), fA.size(), -1, 1, 1);
        fill_random(fB.data(), fB.size(), -1, 1, 2);
        std::vector<half> hA(fA.size()), hB(fB.size());
        for (size_t i = 0; i < hA.size(); i++) { hA[i] = __float2half(fA[i]); fA[i] = __half2float(hA[i]); }
        for (size_t i = 0; i < hB.size(); i++) { hB[i] = __float2half(fB[i]); fB[i] = __half2float(hB[i]); }
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                double s = 0;
                for (int k = 0; k < K; k++) s += (double)fA[(size_t)i * K + k] * fB[(size_t)k * N + j];
                ref[(size_t)i * N + j] = (float)s;
            }
        half *dA, *dB; float* dC;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2)); CUDA_CHECK(cudaMalloc(&dB, hB.size() * 2));
        CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
        for (int w = 0; w < 5; w++) {
            CUDA_CHECK(cudaMemset(dC, 0, hC.size() * 4));
            launch(w, dA, dB, dC, M, N, K); CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
            b.verify(hC.data(), ref.data(), hC.size(), nm[w], 1e-2f, 1e-2f);
        }
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    {   // timing
        int M = 4096, N = 4096, K = 4096;
        printf("\n[timing %dx%dx%d, fp16 in / fp32 accumulate]\n", M, N, K);
        half *dA, *dB; float* dC;
        CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * 2)); CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * 2));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
        CUDA_CHECK(cudaMemset(dA, 0, (size_t)M * K * 2)); CUDA_CHECK(cudaMemset(dB, 0, (size_t)K * N * 2));
        double fl = 2.0 * M * N * K;
        std::vector<std::function<void()>> fns;
        for (int w = 0; w < 5; w++) fns.push_back([&, w] { launch(w, dA, dB, dC, M, N, K); });
        std::vector<float> tm = b.time_interleaved(fns, 8, 2);   // interleaved: same thermal state
        for (int w = 0; w < 5; w++)
            printf("  %s  %7.2f ms   %7.0f GFLOP/s   %5.2fx vs D\n", nm[w], tm[w], fl / tm[w] / 1e6, tm[0] / tm[w]);
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }
    cublasDestroy(h);
    return b.finish();
}

// bench.cuh -- single-header roofline + correctness harness
//
//   Bench b;                                   // query device, derive peak BW / FP32
//   float ms = b.time([&]{ kernel<<<g,t>>>(...); });
//   b.report("kernel", ms, bytes_moved, flops_done);
//   b.verify(h_out, h_ref, n);
//
// Every kernel in this repo is reported against the device roofline, so each
// number comes with "how far is this from what the hardware allows".

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n  -> %s\n",               \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_));      \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Call after a launch; otherwise a launch error surfaces at some later API call.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

struct Bench {
    cudaDeviceProp prop;
    double peak_bw_GBs   = 0;   // theoretical DRAM bandwidth
    double peak_fp32_GFs = 0;   // theoretical FP32 (CUDA cores, no Tensor Cores)
    int    n_fail        = 0;

    Bench(int dev = 0) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // CUDA 13 removed clockRate / memoryClockRate from cudaDeviceProp;
        // query them as attributes instead (both in kHz).
        int mem_clk_kHz = 0, sm_clk_kHz = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_kHz, cudaDevAttrMemoryClockRate, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_clk_kHz,  cudaDevAttrClockRate,       dev));

        // BW = mem clock * bus width / 8 * 2 (DDR)
        peak_bw_GBs = 2.0 * mem_clk_kHz * 1000.0 * (prop.memoryBusWidth / 8.0) / 1e9;

        // FP32 = SMs * FP32 cores/SM * 2 (FMA) * SM clock
        int cores_per_sm = fp32_cores_per_sm(prop.major, prop.minor);
        peak_fp32_GFs = (double)prop.multiProcessorCount * cores_per_sm * 2.0
                      * sm_clk_kHz * 1000.0 / 1e9;

        printf("=====================================================================\n");
        printf(" Device : %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
        printf(" SMs    : %d   |  shared/block: %zu KB  |  L2: %d KB\n",
               prop.multiProcessorCount, prop.sharedMemPerBlock / 1024, prop.l2CacheSize / 1024);
        printf(" Peak BW: %.1f GB/s  |  Peak FP32: %.1f GFLOP/s\n", peak_bw_GBs, peak_fp32_GFs);
        printf(" Ridge  : %.2f FLOP/byte  (below this arithmetic intensity -> memory-bound)\n",
               peak_fp32_GFs / peak_bw_GBs);
        printf("=====================================================================\n");
    }

    static int fp32_cores_per_sm(int major, int minor) {
        switch (major) {
            case 7:  return 64;                        // Volta / Turing
            case 8:  return (minor == 0) ? 64 : 128;   // A100 / consumer Ampere, Ada
            case 9:  return 128;                       // Hopper
            case 10:
            case 12: return 128;                       // Blackwell
            default: return 128;
        }
    }

    // Warm up, then take the median of many runs (robust to laptop clock throttling).
    template <typename F>
    float time(F&& launch, int warmup = 10, int iters = 50) {
        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK_KERNEL();

        std::vector<float> samples(iters);
        cudaEvent_t beg, end;
        CUDA_CHECK(cudaEventCreate(&beg));
        CUDA_CHECK(cudaEventCreate(&end));
        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaEventRecord(beg));
            launch();
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            CUDA_CHECK(cudaEventElapsedTime(&samples[i], beg, end));
        }
        CUDA_CHECK(cudaEventDestroy(beg));
        CUDA_CHECK(cudaEventDestroy(end));
        std::sort(samples.begin(), samples.end());
        return samples[iters / 2];
    }

    // Like time(), but writes a buffer twice the L2 size before every timed launch, so the
    // kernel's inputs come from DRAM. Needed for decode-style kernels whose weights or KV
    // (tens of MB) would otherwise stay L2-resident across back-to-back runs; in a real
    // model the other layers evict them between uses.
    template <typename F>
    float time_cold(F&& launch, int warmup = 5, int iters = 30) {
        ensure_flush();
        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK_KERNEL();
        std::vector<float> samples(iters);
        cudaEvent_t beg, end;
        CUDA_CHECK(cudaEventCreate(&beg));
        CUDA_CHECK(cudaEventCreate(&end));
        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaMemsetAsync(flush_buf, i & 0xff, flush_bytes));
            CUDA_CHECK(cudaEventRecord(beg));
            launch();
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            CUDA_CHECK(cudaEventElapsedTime(&samples[i], beg, end));
        }
        CUDA_CHECK(cudaEventDestroy(beg));
        CUDA_CHECK(cudaEventDestroy(end));
        std::sort(samples.begin(), samples.end());
        return samples[iters / 2];
    }
    void*  flush_buf   = nullptr;
    size_t flush_bytes = 0;
    void ensure_flush() {
        if (flush_buf) return;
        flush_bytes = std::max<size_t>((size_t)2 * prop.l2CacheSize, (size_t)64 << 20);
        CUDA_CHECK(cudaMalloc(&flush_buf, flush_bytes));
    }

    // Times several alternatives in interleaved rounds (A B C A B C ...) and returns each
    // one's median. A laptop GPU's clock drifts with temperature over a run; timing methods
    // back to back would hand whichever runs first the cooler, faster GPU. Interleaving gives
    // every method the same conditions, so the RATIOS between methods are trustworthy.
    std::vector<float> time_interleaved(const std::vector<std::function<void()>>& fns,
                                        int rounds = 12, int per_round = 3, int warmup = 3,
                                        bool cold = false) {      // cold: flush L2 before each sample
        if (cold) ensure_flush();
        for (auto& fn : fns)
            for (int i = 0; i < warmup; ++i) fn();
        CUDA_CHECK_KERNEL();
        std::vector<std::vector<float>> samples(fns.size());
        cudaEvent_t beg, end;
        CUDA_CHECK(cudaEventCreate(&beg));
        CUDA_CHECK(cudaEventCreate(&end));
        for (int r = 0; r < rounds; ++r)
            for (size_t m = 0; m < fns.size(); ++m)
                for (int i = 0; i < per_round; ++i) {
                    float ms;
                    if (cold) CUDA_CHECK(cudaMemsetAsync(flush_buf, (r + i) & 0xff, flush_bytes));
                    CUDA_CHECK(cudaEventRecord(beg));
                    fns[m]();
                    CUDA_CHECK(cudaEventRecord(end));
                    CUDA_CHECK(cudaEventSynchronize(end));
                    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
                    samples[m].push_back(ms);
                }
        CUDA_CHECK(cudaEventDestroy(beg));
        CUDA_CHECK(cudaEventDestroy(end));
        std::vector<float> med;
        for (auto& s : samples) {
            std::sort(s.begin(), s.end());
            med.push_back(s[s.size() / 2]);
        }
        return med;
    }

    void header() {
        printf("%-34s %9s %9s %8s %10s %8s  %s\n",
               "kernel", "time(ms)", "GB/s", "%BW", "GFLOP/s", "%FP32", "bound");
        printf("---------------------------------------------------------------------\n");
    }

    // bytes: compulsory traffic (read + write, no re-reads). flops: 0 for pure data movement.
    void report(const char* name, float ms, double bytes, double flops = 0) {
        double s   = ms / 1e3;
        double gbs = bytes / s / 1e9;
        double gfs = flops / s / 1e9;
        double ai  = (bytes > 0) ? flops / bytes : 0;
        double ridge = peak_fp32_GFs / peak_bw_GBs;
        const char* bound = (flops == 0 || ai < ridge) ? "mem" : "compute";
        printf("%-34s %9.4f %9.1f %7.1f%% %10.1f %7.1f%%  %s (AI=%.2f)\n",
               name, ms, gbs, 100.0 * gbs / peak_bw_GBs,
               gfs, (peak_fp32_GFs > 0 ? 100.0 * gfs / peak_fp32_GFs : 0.0), bound, ai);
    }

    // Mixed abs/rel tolerance: different reduction orders are not bit-exact, and that is expected.
    bool verify(const float* got, const float* ref, size_t n,
                const char* tag = "", float rtol = 1e-4f, float atol = 1e-5f) {
        size_t bad = 0, first_bad = 0;
        double max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(got[i]) != std::isnan(ref[i]) || std::isinf(got[i]) != std::isinf(ref[i])) {
                if (!bad) first_bad = i;
                ++bad;
                continue;
            }
            double a = std::fabs((double)got[i] - (double)ref[i]);
            double r = a / (std::fabs((double)ref[i]) + 1e-12);
            max_abs = std::max(max_abs, a);
            max_rel = std::max(max_rel, r);
            if (a > atol + rtol * std::fabs((double)ref[i])) {
                if (!bad) first_bad = i;
                ++bad;
            }
        }
        if (bad) {
            printf("  [FAIL] %s  %zu/%zu mismatched  max_abs=%.3e max_rel=%.3e\n",
                   tag, bad, n, max_abs, max_rel);
            printf("         first bad @ %zu: got=%.8f ref=%.8f\n", first_bad, got[first_bad], ref[first_bad]);
            ++n_fail;
            return false;
        }
        printf("  [ok]   %s  max_abs=%.3e max_rel=%.3e\n", tag, max_abs, max_rel);
        return true;
    }

    int finish() {
        printf("---------------------------------------------------------------------\n");
        if (n_fail) printf("  %d CHECK(S) FAILED\n", n_fail);
        else        printf("  all checks passed\n");
        return n_fail ? 1 : 0;
    }
};

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline void fill_random(float* p, size_t n, float lo = -1.f, float hi = 1.f, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) p[i] = lo + (hi - lo) * (rand() / (float)RAND_MAX);
}

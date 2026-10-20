// 08_reduction_race.cu
//
// A shared-memory race that tests cannot find but compute-sanitizer can.
//
// A block reduction usually ends with
//     __syncthreads();  return s[0];
// That is correct for ONE call. Called twice in a row (LayerNorm: mean, then variance),
// both inlined copies use the same __shared__ s[32] (nvcc merges them; -Xptxas -v shows a
// single 128-byte allocation). A fast warp can finish the first call, start the second,
// and write s[warp] before a slow warp has read s[0] from the first call.
//
// Fix: read into a register, then barrier, then return.
//
// Run:
//   compute-sanitizer --tool racecheck bin/08_reduction_race   (reports hazards in v1)
// Measured on RTX 5060 Laptop: 200 normal runs of the unsafe version give 0 wrong rows.
// Under racecheck (which changes warp scheduling) the same binary reports 13,732 hazards,
// all in block_sum_unsafe, and 1,333 / 2,048 rows come out wrong. The safe version: 0 and 0.

#include "common/bench.cuh"

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float block_sum_unsafe(float v) {
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
    return s[0];                    // no barrier after this read
}

__device__ __forceinline__ float block_sum_safe(float v) {
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
    __syncthreads();                // nobody may overwrite s until everyone has read s[0]
    return r;
}

template <bool SAFE>
__global__ void layernorm_stats(const float* x, float* mean_out, float* var_out, int N) {
    const float* xr = x + (size_t)blockIdx.x * N;
    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += xr[i];
    float mean = (SAFE ? block_sum_safe(s) : block_sum_unsafe(s)) / N;     // call 1
    float q = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) { float d = xr[i] - mean; q += d * d; }
    float var = (SAFE ? block_sum_safe(q) : block_sum_unsafe(q)) / N;       // call 2
    if (threadIdx.x == 0) { mean_out[blockIdx.x] = mean; var_out[blockIdx.x] = var; }
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200;      // pass 1 when running under racecheck
    int rows = 2048, N = 1024, T = 256;
    size_t n = (size_t)rows * N;
    std::vector<float> hx(n), hm(rows), hv(rows), rv(rows);
    fill_random(hx.data(), n, -2.f, 2.f, 1);
    for (int r = 0; r < rows; r++) {
        double s = 0, q = 0;
        for (int i = 0; i < N; i++) s += hx[(size_t)r * N + i];
        double mu = s / N;
        for (int i = 0; i < N; i++) { double d = hx[(size_t)r * N + i] - mu; q += d * d; }
        rv[r] = (float)(q / N);
    }
    float *dx, *dm, *dv;
    CUDA_CHECK(cudaMalloc(&dx, n * 4)); CUDA_CHECK(cudaMalloc(&dm, rows * 4)); CUDA_CHECK(cudaMalloc(&dv, rows * 4));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));

    printf("\n[LayerNorm statistics, %d runs each: does the unsafe version ever give a wrong number?]\n", iters);
    for (int v = 0; v < 2; v++) {
        int worst = 0;
        for (int it = 0; it < iters; it++) {
            if (v == 0) layernorm_stats<false><<<rows, T>>>(dx, dm, dv, N);
            else        layernorm_stats<true ><<<rows, T>>>(dx, dm, dv, N);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(hv.data(), dv, rows * 4, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int r = 0; r < rows; r++) bad += (fabs(hv[r] - rv[r]) / (fabs(rv[r]) + 1e-9) > 1e-3);
            worst = std::max(worst, bad);
        }
        printf("  %s worst run: %d / %d rows wrong\n", v == 0 ? "unsafe (2 barriers)" : "safe   (3 barriers)", worst, rows);
    }
    printf("\n  A clean test run proves nothing here. Check with:\n");
    printf("    compute-sanitizer --tool racecheck bin/08_reduction_race 1\n");
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dm)); CUDA_CHECK(cudaFree(dv));
    return 0;
}

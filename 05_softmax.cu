// 05_softmax.cu
//
// Row softmax three ways, one block per row:
//   safe    3 passes: max, sum(exp(x - max)), write
//   online  2 passes: running (max, sum) in one pass, then write. The (m, d) merge rule
//           d = d1 * exp(m1 - m) + d2 * exp(m2 - m) is the same one used to merge split-KV
//           partials in 01_decode_attention.cu and inside FlashAttention.
//   vec4    safe softmax with float4 loads/stores (used only when N % 4 == 0)
//
// Tested on shapes that break naive kernels: N not a multiple of the block size, N = 1,
// rows much longer than the block, and an input with a +500 outlier (overflows expf
// without max subtraction).

#include "common/bench.cuh"

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Two-level block reduction: warp shuffle, then warp 0 reduces the per-warp results.
// The trailing __syncthreads matters: without it, a second call in the same kernel can
// overwrite s[0] before every thread has read it (see 08_reduction_race.cu).
__device__ __forceinline__ float block_max(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    v = warp_max(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < nw) ? s[threadIdx.x] : -INFINITY;   // neutral element for max
        v = warp_max(v);
        if (threadIdx.x == 0) s[0] = v;
    }
    __syncthreads();
    float r = s[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    v = warp_sum(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (threadIdx.x < nw) ? s[threadIdx.x] : 0.0f;        // neutral element for sum
        v = warp_sum(v);
        if (threadIdx.x == 0) s[0] = v;
    }
    __syncthreads();
    float r = s[0];
    __syncthreads();
    return r;
}

__global__ void softmax_safe(const float* x, float* y, int N) {
    const float* xr = x + (size_t)blockIdx.x * N;
    float*       yr = y + (size_t)blockIdx.x * N;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = block_max(m);
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += expf(xr[i] - m);
    s = block_sum(s);
    for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = expf(xr[i] - m) / s;
}

// Merge two (max, sum-of-exp) states. Either side may be empty (m = -inf).
__device__ __forceinline__ void merge_md(float& m, float& d, float m2, float d2) {
    if (m2 == -INFINITY) return;
    if (m  == -INFINITY) { m = m2; d = d2; return; }
    float mn = fmaxf(m, m2);
    d = d * expf(m - mn) + d2 * expf(m2 - mn);
    m = mn;
}

__device__ __forceinline__ void block_merge(float& m, float& d) {
    __shared__ float sm[32], sd[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o /= 2) {
        float m2 = __shfl_down_sync(0xffffffff, m, o);
        float d2 = __shfl_down_sync(0xffffffff, d, o);
        merge_md(m, d, m2, d2);
    }
    if (lane == 0) { sm[wid] = m; sd[wid] = d; }
    __syncthreads();
    if (wid == 0) {
        m = (lane < nw) ? sm[lane] : -INFINITY;
        d = (lane < nw) ? sd[lane] : 0.0f;
        for (int o = 16; o > 0; o /= 2) {
            float m2 = __shfl_down_sync(0xffffffff, m, o);
            float d2 = __shfl_down_sync(0xffffffff, d, o);
            merge_md(m, d, m2, d2);
        }
        if (lane == 0) { sm[0] = m; sd[0] = d; }
    }
    __syncthreads();
    m = sm[0];
    d = sd[0];
}

__global__ void softmax_online(const float* x, float* y, int N) {
    const float* xr = x + (size_t)blockIdx.x * N;
    float*       yr = y + (size_t)blockIdx.x * N;
    float m = -INFINITY, d = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = xr[i];
        float mn = fmaxf(m, v);
        d = d * expf(m - mn) + expf(v - mn);       // rescale the old sum to the new max
        m = mn;
    }
    block_merge(m, d);
    for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = expf(xr[i] - m) / d;
}

// Requires N % 4 == 0 so every row starts 16-byte aligned.
__global__ void softmax_vec4(const float* x, float* y, int N) {
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * N);
    float4*       y4 = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * N);
    int n4 = N / 4;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = x4[i];
        m = fmaxf(fmaxf(m, v.x), fmaxf(fmaxf(v.y, v.z), v.w));
    }
    m = block_max(m);
    float s = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = x4[i];
        s += expf(v.x - m) + expf(v.y - m) + expf(v.z - m) + expf(v.w - m);
    }
    s = block_sum(s);
    float inv = 1.0f / s;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = x4[i], r;
        r.x = expf(v.x - m) * inv; r.y = expf(v.y - m) * inv;
        r.z = expf(v.z - m) * inv; r.w = expf(v.w - m) * inv;
        y4[i] = r;
    }
}

void cpu_softmax(const float* x, float* y, int rows, int N) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * N;
        float* yr = y + (size_t)r * N;
        double m = -1e300, s = 0;
        for (int i = 0; i < N; i++) m = std::max(m, (double)xr[i]);
        for (int i = 0; i < N; i++) s += exp(xr[i] - m);
        for (int i = 0; i < N; i++) yr[i] = (float)(exp(xr[i] - m) / s);
    }
}

int main() {
    Bench b;
    const int T = 256;
    struct Case { int rows, N; const char* tag; };
    Case cases[] = {{512, 1000, "N not a multiple of 256"},
                    {64,  5000, "row much longer than block"},
                    {1000,   1, "N = 1"},
                    {8,  70000, "very long row"},
                    {16,  1001, "N odd (vec4 skipped)"}};
    const char* nm[3] = {"safe  ", "online", "vec4  "};

    printf("\n[correctness vs fp64 reference, row 0 contains a +500 outlier]\n");
    for (auto& c : cases) {
        size_t n = (size_t)c.rows * c.N;
        std::vector<float> hx(n), hy(n), ref(n);
        fill_random(hx.data(), n, -10.f, 10.f, 42);
        if (c.N > 2) hx[0] = 500.0f;
        cpu_softmax(hx.data(), ref.data(), c.rows, c.N);
        float *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dx, n * 4));
        CUDA_CHECK(cudaMalloc(&dy, n * 4));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * 4, cudaMemcpyHostToDevice));
        for (int v = 0; v < 3; v++) {
            if (v == 2 && c.N % 4) continue;
            CUDA_CHECK(cudaMemset(dy, 0, n * 4));
            if (v == 0) softmax_safe  <<<c.rows, T>>>(dx, dy, c.N);
            if (v == 1) softmax_online<<<c.rows, T>>>(dx, dy, c.N);
            if (v == 2) softmax_vec4  <<<c.rows, T>>>(dx, dy, c.N);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * 4, cudaMemcpyDeviceToHost));
            char tag[96];
            snprintf(tag, sizeof tag, "%s rows=%-4d N=%-6d %s", nm[v], c.rows, c.N, c.tag);
            b.verify(hy.data(), ref.data(), n, tag, 1e-4f, 1e-6f);
        }
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
    }

    printf("\n[timing 4096 x 4096 fp32; ideal traffic = read once + write once]\n");
    b.header();
    int rows = 4096, N = 4096;
    size_t n = (size_t)rows * N;
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * 4));
    CUDA_CHECK(cudaMalloc(&dy, n * 4));
    CUDA_CHECK(cudaMemset(dx, 0, n * 4));
    double bytes = 2.0 * n * 4;
    b.report("softmax_safe   (3 passes)", b.time([&] { softmax_safe  <<<rows, T>>>(dx, dy, N); }), bytes);
    b.report("softmax_online (2 passes)", b.time([&] { softmax_online<<<rows, T>>>(dx, dy, N); }), bytes);
    b.report("softmax_vec4   (3 passes)", b.time([&] { softmax_vec4  <<<rows, T>>>(dx, dy, N); }), bytes);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return b.finish();
}

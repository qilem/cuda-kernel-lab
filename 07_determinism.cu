// 07_determinism.cu
//
// Floating-point addition is not associative, so any reduction whose order depends on
// scheduling (float atomicAdd) gives different bits from run to run. Two places this shows
// up, each with a deterministic replacement:
//
//   1. full-array sum      atomicAdd per element / per block   vs   two-pass partial sums
//   2. scatter_add         out[idx[i]] += src[i] with atomics  vs   stable sort by idx +
//      (embedding grads,   segmented sum (one block owns one output row, fixed order)
//       MoE combine)
//
// Determinism is measured bit-for-bit across repeated runs, not with a tolerance.

#include "common/bench.cuh"
#include <cstring>

unsigned bits(float f) { unsigned u; memcpy(&u, &f, 4); return u; }

__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ float block_sum(float v) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    v = warp_sum(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? s[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    v = s[0];
    __syncthreads();
    return v;
}

// ---------------------------------------------------------------- 1. sum
__global__ void sum_atomic_each(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, x[i]);
}
__global__ void sum_block_atomic(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = block_sum((i < n) ? x[i] : 0.0f);
    if (threadIdx.x == 0) atomicAdd(out, v);
}
__global__ void sum_partial(const float* x, float* partial, int n) {   // pass 1: one slot per block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = block_sum((i < n) ? x[i] : 0.0f);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}
__global__ void sum_final(const float* partial, float* out, int m) {   // pass 2: one block, fixed order
    float v = 0.0f;
    for (int i = threadIdx.x; i < m; i += blockDim.x) v += partial[i];
    v = block_sum(v);
    if (threadIdx.x == 0) out[0] = v;
}

// ---------------------------------------------------------------- 2. scatter_add
__global__ void scatter_atomic(const float* src, const int* idx, float* out, int N, int D) {
    int i = blockIdx.x;                                 // one block per source row
    int row = idx[i];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        atomicAdd(&out[(size_t)row * D + d], src[(size_t)i * D + d]);
}

__device__ __forceinline__ int lower_bound(const int* a, int n, int v) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// keys: idx after a STABLE sort; perm[k]: which source row is k-th after sorting.
// One block owns one output row, so no two threads ever write the same address and the
// summation order is fixed by the stable sort.
__global__ void scatter_sorted(const float* src, const int* keys, const int* perm, float* out, int N, int D) {
    int j = blockIdx.x;
    int lo = lower_bound(keys, N, j);                   // keys[lo, hi) == j
    int hi = lower_bound(keys, N, j + 1);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int k = lo; k < hi; k++) acc += src[(size_t)perm[k] * D + d];
        out[(size_t)j * D + d] = acc;                   // empty segment writes 0: no memset needed
    }
}

int main() {
    Bench b;

    printf("\n[float addition is not associative]\n");
    volatile float a = 1e8f, one = 1.0f, c = -1e8f;
    printf("  (1e8 + 1) - 1e8 = %g      (1e8 - 1e8) + 1 = %g\n", (a + one) + c, (a + c) + one);

    // ---------------- 1. sum
    {
        const int n = 1 << 22, T = 256, blocks = (n + T - 1) / T;
        std::vector<float> hx(n);
        fill_random(hx.data(), n, 0.0f, 1.0f, 11);
        double ref = 0;
        for (int i = 0; i < n; i++) ref += hx[i];
        float *dx, *dout, *dpart;
        CUDA_CHECK(cudaMalloc(&dx, (size_t)n * 4));
        CUDA_CHECK(cudaMalloc(&dout, 4));
        CUDA_CHECK(cudaMalloc(&dpart, blocks * 4));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
        printf("\n[sum of %d floats, 5 runs each; fp64 reference %.4f]\n", n, ref);
        const char* names[3] = {"atomicAdd per element", "atomicAdd per block  ", "two-pass, no atomics "};
        for (int method = 0; method < 3; method++) {
            unsigned first = 0;
            int distinct = 1;
            std::vector<unsigned> seen;
            float val = 0;
            for (int run = 0; run < 5; run++) {
                CUDA_CHECK(cudaMemset(dout, 0, 4));
                if (method == 0) sum_atomic_each<<<blocks, T>>>(dx, dout, n);
                if (method == 1) sum_block_atomic<<<blocks, T>>>(dx, dout, n);
                if (method == 2) { sum_partial<<<blocks, T>>>(dx, dpart, n); sum_final<<<1, 1024>>>(dpart, dout, blocks); }
                CUDA_CHECK_KERNEL();
                CUDA_CHECK(cudaMemcpy(&val, dout, 4, cudaMemcpyDeviceToHost));
                unsigned u = bits(val);
                if (run == 0) first = u;
                if (std::find(seen.begin(), seen.end(), u) == seen.end()) seen.push_back(u);
            }
            distinct = (int)seen.size();
            printf("  %s  last=%.4f  distinct bit patterns over 5 runs: %d\n", names[method], val, distinct);
            if (method == 2 && distinct != 1) b.n_fail++;
            (void)first;
        }
        CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(dpart));
    }

    // ---------------- 2. scatter_add
    {
        int N = 200000, D = 64, M = 1000;
        printf("\n[scatter_add: %d rows of width %d into %d output rows]\n", N, D, M);
        std::vector<float> hsrc((size_t)N * D), h1((size_t)M * D), h2((size_t)M * D), ref((size_t)M * D);
        std::vector<int> hidx(N);
        srand(3);
        for (auto& v : hsrc) v = rand() / (float)RAND_MAX - 0.5f;
        for (int i = 0; i < N; i++) hidx[i] = rand() % M;
        std::vector<double> acc((size_t)M * D, 0.0);
        for (int i = 0; i < N; i++)
            for (int d = 0; d < D; d++) acc[(size_t)hidx[i] * D + d] += hsrc[(size_t)i * D + d];
        for (size_t t = 0; t < acc.size(); t++) ref[t] = (float)acc[t];

        std::vector<int> ord(N), keys(N), perm(N);       // stable sort (host here; cub on device in practice)
        for (int i = 0; i < N; i++) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return hidx[x] < hidx[y]; });
        for (int k = 0; k < N; k++) { perm[k] = ord[k]; keys[k] = hidx[ord[k]]; }

        float *dsrc, *dout;
        int *didx, *dkeys, *dperm;
        CUDA_CHECK(cudaMalloc(&dsrc, hsrc.size() * 4)); CUDA_CHECK(cudaMalloc(&dout, (size_t)M * D * 4));
        CUDA_CHECK(cudaMalloc(&didx, N * 4)); CUDA_CHECK(cudaMalloc(&dkeys, N * 4)); CUDA_CHECK(cudaMalloc(&dperm, N * 4));
        CUDA_CHECK(cudaMemcpy(dsrc, hsrc.data(), hsrc.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(didx, hidx.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dkeys, keys.data(), N * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dperm, perm.data(), N * 4, cudaMemcpyHostToDevice));
        size_t ob = (size_t)M * D * 4;

        for (int v = 0; v < 2; v++) {
            auto go = [&] {
                if (v == 0) { cudaMemsetAsync(dout, 0, ob); scatter_atomic<<<N, 64>>>(dsrc, didx, dout, N, D); }
                else        scatter_sorted<<<M, 64>>>(dsrc, dkeys, dperm, dout, N, D);
            };
            for (int run = 0; run < 2; run++) {
                if (v == 1) CUDA_CHECK(cudaMemset(dout, 0xFF, ob));   // NaN fill: every slot must be written
                go();
                CUDA_CHECK_KERNEL();
                CUDA_CHECK(cudaMemcpy(run == 0 ? h1.data() : h2.data(), dout, ob, cudaMemcpyDeviceToHost));
            }
            int diff = 0;
            for (size_t t = 0; t < h1.size(); t++) diff += (bits(h1[t]) != bits(h2[t]));
            char tag[96];
            snprintf(tag, sizeof tag, "%s run-to-run bit diffs %5d / %zu",
                     v == 0 ? "atomicAdd          " : "sort + segment sum ", diff, h1.size());
            b.verify(h1.data(), ref.data(), h1.size(), tag, 1e-4f, 1e-4f);
            float ms = b.time(go, 5, 20);
            printf("         time %.4f ms\n", ms);
            if (v == 1 && diff) b.n_fail++;
        }
        printf("  (sort cost is excluded: the index set is reused across calls in this benchmark)\n");
        CUDA_CHECK(cudaFree(dsrc)); CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(didx));
        CUDA_CHECK(cudaFree(dkeys)); CUDA_CHECK(cudaFree(dperm));
    }
    return b.finish();
}

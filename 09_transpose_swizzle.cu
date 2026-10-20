// 09_transpose_swizzle.cu
//
// Matrix transpose through shared memory: three ways to deal with bank conflicts.
//
//   naive    read coalesced, write strided by R floats (every write its own sector)
//   padded   tile[32][33]: the extra column shifts each row by one bank. Works, but costs
//            memory and breaks the 128-byte row alignment that TMA / ldmatrix-style
//            loads require.
//   XOR      tile[32][32], logical (r, c) stored at physical (r, c ^ r). The map is a
//            bijection within each row (no extra memory), keeps rows 128-byte aligned, and
//            a column read tile[x][y ^ x] for x = 0..31 lands on 32 distinct banks.
//            Hopper's TMA swizzle modes are defined as this kind of address-bit XOR.

#include "common/bench.cuh"

#define TILE 32

__global__ void transpose_naive(const float* in, float* out, int R, int C) {
    int c = blockIdx.x * TILE + threadIdx.x;
    int r = blockIdx.y * TILE + threadIdx.y;
    if (r < R && c < C) out[(size_t)c * R + r] = in[(size_t)r * C + c];
}

__global__ void transpose_padded(const float* in, float* out, int R, int C) {
    __shared__ float tile[TILE][TILE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int c = blockIdx.x * TILE + tx, r = blockIdx.y * TILE + ty;
    if (r < R && c < C) tile[ty][tx] = in[(size_t)r * C + c];
    __syncthreads();
    int orow = blockIdx.x * TILE + ty, ocol = blockIdx.y * TILE + tx;   // block coords swap
    if (orow < C && ocol < R) out[(size_t)orow * R + ocol] = tile[tx][ty];
}

__global__ void transpose_xor(const float* in, float* out, int R, int C) {
    __shared__ float tile[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int c = blockIdx.x * TILE + tx, r = blockIdx.y * TILE + ty;
    // tx is always the fastest index on the global side (coalesced read AND write);
    // logical (ty, tx) is stored at physical column tx ^ ty
    if (r < R && c < C) tile[ty][tx ^ ty] = in[(size_t)r * C + c];
    __syncthreads();
    int orow = blockIdx.x * TILE + ty, ocol = blockIdx.y * TILE + tx;
    // this thread needs logical (tx, ty) -> physical column ty ^ tx
    if (orow < C && ocol < R) out[(size_t)orow * R + ocol] = tile[tx][ty ^ tx];
}

int main() {
    Bench b;
    for (auto sh : {std::pair<int, int>{1000, 777}, {4096, 4096}}) {
        int R = sh.first, C = sh.second;
        size_t n = (size_t)R * C;
        std::vector<float> hin(n), hout(n), ref(n);
        for (size_t i = 0; i < n; i++) hin[i] = (float)(i % 10007);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) ref[(size_t)c * R + r] = hin[(size_t)r * C + c];
        float *din, *dout;
        CUDA_CHECK(cudaMalloc(&din, n * 4));
        CUDA_CHECK(cudaMalloc(&dout, n * 4));
        CUDA_CHECK(cudaMemcpy(din, hin.data(), n * 4, cudaMemcpyHostToDevice));
        dim3 blk(TILE, TILE), grd((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
        printf("\n[R=%d C=%d]\n", R, C);
        const char* nm[3] = {"naive            ", "padded tile[32][33]", "XOR swizzle      "};
        for (int v = 0; v < 3; v++) {
            auto go = [&] {
                if (v == 0) transpose_naive <<<grd, blk>>>(din, dout, R, C);
                if (v == 1) transpose_padded<<<grd, blk>>>(din, dout, R, C);
                if (v == 2) transpose_xor   <<<grd, blk>>>(din, dout, R, C);
            };
            CUDA_CHECK(cudaMemset(dout, 0, n * 4));
            go();
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * 4, cudaMemcpyDeviceToHost));
            b.verify(hout.data(), ref.data(), n, nm[v], 0.f, 0.f);
            if (n > 1000000) {
                if (v == 0) b.header();
                b.report(nm[v], b.time(go), 2.0 * n * 4);
            }
        }
        CUDA_CHECK(cudaFree(din));
        CUDA_CHECK(cudaFree(dout));
    }
    return b.finish();
}

// 04_mma_ptx.cu
//
// Hand-written inline PTX mma.sync.aligned.m16n8k16 vs the WMMA API.
//
// WMMA only exposes 16x16x16 (for fp16), but the hardware instruction on Ampere+ is
// m16n8k16. Writing the PTX directly means placing every A/B/C element in exactly the
// register and lane the PTX ISA specifies:
//
//   lane = 4 * groupID + threadID_in_group      (groupID 0..7, tig 0..3)
//   A (16x16, row): a0,a1 = A[g][2t..2t+1]     a2,a3 = A[g+8][2t..2t+1]
//                   a4,a5 = A[g][2t+8..2t+9]   a6,a7 = A[g+8][2t+8..2t+9]
//   B (16x8, col) : b0,b1 = B[2t..2t+1][g]     b2,b3 = B[2t+8..2t+9][g]
//   D (16x8)      : d0,d1 = D[g][2t..2t+1]     d2,d3 = D[g+8][2t..2t+1]
//
// The example multiplies a 16x16 A by a 16x8 B. One mma.sync does it exactly; WMMA has to
// pad B to 16 columns, so half of its MACs produce zeros.

#include "common/bench.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>
using namespace nvcuda;

__global__ void mma_ptx(const half* A, const half* B, float* D) {
    int lane = threadIdx.x;
    int g = lane / 4;          // groupID
    int t = lane % 4;          // thread id in group

    half a[8];
    a[0] = A[(g    ) * 16 + t * 2 + 0];
    a[1] = A[(g    ) * 16 + t * 2 + 1];
    a[2] = A[(g + 8) * 16 + t * 2 + 0];
    a[3] = A[(g + 8) * 16 + t * 2 + 1];
    a[4] = A[(g    ) * 16 + t * 2 + 8];
    a[5] = A[(g    ) * 16 + t * 2 + 9];
    a[6] = A[(g + 8) * 16 + t * 2 + 8];
    a[7] = A[(g + 8) * 16 + t * 2 + 9];

    half b[4];
    b[0] = B[(t * 2 + 0) * 8 + g];
    b[1] = B[(t * 2 + 1) * 8 + g];
    b[2] = B[(t * 2 + 8) * 8 + g];
    b[3] = B[(t * 2 + 9) * 8 + g];

    const uint32_t* ra = reinterpret_cast<const uint32_t*>(a);   // two halves per 32-bit register
    const uint32_t* rb = reinterpret_cast<const uint32_t*>(b);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1]),
          "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]));

    D[(g    ) * 8 + t * 2 + 0] = d[0];
    D[(g    ) * 8 + t * 2 + 1] = d[1];
    D[(g + 8) * 8 + t * 2 + 0] = d[2];
    D[(g + 8) * 8 + t * 2 + 1] = d[3];
}

// WMMA: only 16x16x16, so B must be padded to 16 columns and so must the output.
__global__ void wmma_16x16(const half* A, const half* B16, float* D16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B16, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(D16, c, 16, wmma::mem_row_major);
}

int main() {
    std::vector<half> hA(16 * 16), hB(16 * 8), hB16(16 * 16, __float2half(0.f));
    std::vector<float> ref(16 * 8, 0.f);
    srand(7);
    for (auto& x : hA) x = __float2half((float)(rand() % 7 - 3));   // small ints: exact in fp16
    for (auto& x : hB) x = __float2half((float)(rand() % 7 - 3));
    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 8; n++) hB16[k * 16 + n] = hB[k * 8 + n];
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) {
            float s = 0;
            for (int k = 0; k < 16; k++) s += __half2float(hA[m * 16 + k]) * __half2float(hB[k * 8 + n]);
            ref[m * 8 + n] = s;
        }

    half *dA, *dB, *dB16;
    float *dD, *dD16;
    CUDA_CHECK(cudaMalloc(&dA, 16 * 16 * 2));
    CUDA_CHECK(cudaMalloc(&dB, 16 * 8 * 2));
    CUDA_CHECK(cudaMalloc(&dB16, 16 * 16 * 2));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMalloc(&dD16, 16 * 16 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), 16 * 16 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), 16 * 8 * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB16, hB16.data(), 16 * 16 * 2, cudaMemcpyHostToDevice));

    printf("\n[inline PTX mma.sync.m16n8k16]\n");
    mma_ptx<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    std::vector<float> got(16 * 8);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, 16 * 8 * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += (got[i] != ref[i]);      // integer inputs: exact
    printf("  mismatches vs CPU reference: %d / 128\n", bad);

    printf("\n[WMMA 16x16x16 with B zero-padded to 16 columns]\n");
    wmma_16x16<<<1, 32>>>(dA, dB16, dD16);
    CUDA_CHECK_KERNEL();
    std::vector<float> got16(16 * 16);
    CUDA_CHECK(cudaMemcpy(got16.data(), dD16, 16 * 16 * 4, cudaMemcpyDeviceToHost));
    int bad2 = 0;
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++) bad2 += (got16[m * 16 + n] != ref[m * 8 + n]);
    printf("  mismatches in the 8 useful columns: %d / 128\n", bad2);
    printf("  the other 8 columns are padding: 50%% of the MACs are wasted\n");

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dB16));
    CUDA_CHECK(cudaFree(dD)); CUDA_CHECK(cudaFree(dD16));
    printf("\n  %s\n", (bad == 0 && bad2 == 0) ? "all checks passed" : "CHECK FAILED");
    return (bad == 0 && bad2 == 0) ? 0 : 1;
}

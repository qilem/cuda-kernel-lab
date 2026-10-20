"""Library baselines on the same GPU, same shapes, same cold-L2 timing as the CUDA binaries.

  A. cuDNN fused attention (PyTorch SDPA, CUDNN backend) for GQA decode, dense KV
  B. cuDNN on the shared-prefix batches of 01 (dense per-request KV, so nothing is shared)
  C. tinygemm int4 weight-only GEMM (torch._weight_int4pack_mm, the kernel behind gpt-fast /
     torchao int4), one call per active expert, using the exact per-expert row counts that
     03_moe_wq_decode exported to results/moe_decode_counts.txt

Run from the repo root after bin/03_moe_wq_decode:  python tools/torch_baselines.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DEV = "cuda"
FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)   # 2x the 32 MB L2


def time_cold(fn, warmup=5, iters=30):
    """Median ms over iters; L2 is overwritten before every timed call (as in bench.cuh)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        FLUSH.fill_(i & 0xFF)
        b = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        b.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(b.elapsed_time(e))
    ts.sort()
    return ts[iters // 2]


def cudnn_decode(B, L, Hq=32, Hkv=8, D=128):
    q = torch.randn(B, Hq, 1, D, device=DEV, dtype=torch.float16)
    k = torch.randn(B, Hkv, L, D, device=DEV, dtype=torch.float16)
    v = torch.randn(B, Hkv, L, D, device=DEV, dtype=torch.float16)
    with sdpa_kernel(SDPBackend.MATH):                      # fp32 reference for one sequence
        ref = F.scaled_dot_product_attention(q[:1].float(), k[:1].float(), v[:1].float(), enable_gqa=True)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        err = (out[:1].float() - ref).abs().max().item()
        ms = time_cold(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
    del q, k, v
    torch.cuda.empty_cache()
    return ms, 2.0 * B * Hkv * L * D * 2, err


def main():
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

    print("\n[A] cuDNN fused attention, GQA decode (Hq=32, Hkv=8, D=128, fp16, dense contiguous KV)")
    print(f"  {'shape':<28}{'time(ms)':>10}{'KV GB/s':>10}{'max err':>10}")
    for B, L in [(1, 32768), (8, 8192), (64, 1024)]:
        ms, by, err = cudnn_decode(B, L)
        print(f"  {f'B={B} L={L}':<28}{ms:>10.4f}{by / ms / 1e6:>10.1f}{err:>10.1e}")

    print("\n[B] cuDNN on the shared-prefix batches (each request's full KV, nothing shared)")
    print(f"  {'shape':<38}{'time(ms)':>10}{'KV GB/s':>10}")
    for B, Lp, Ls in [(64, 4096, 256), (32, 16384, 512), (8, 16384, 512)]:
        ms, by, _ = cudnn_decode(B, Lp + Ls)
        print(f"  {f'B={B} prefix={Lp} suffix={Ls}':<38}{ms:>10.4f}{by / ms / 1e6:>10.1f}")

    print("\n[C] tinygemm int4 weight-only GEMM (bf16 activations, group 128), one call per active expert,")
    print("    replayed from a CUDA Graph (no Python / launch gaps)")
    try:
        cfgs = [list(map(int, l.split())) for l in open("results/moe_decode_counts.txt") if l.strip()]
    except FileNotFoundError:
        print("  results/moe_decode_counts.txt missing: run bin/03_moe_wq_decode first")
        return
    print(f"  {'shape':<34}{'experts':>8}{'time(ms)':>10}{'weight MB':>11}{'GB/s':>8}")
    for row in cfgs:
        T, E, d, f, counts = row[0], row[1], row[2], row[3], row[4:]
        active = [n for n in counts if n > 0]
        ws, xs, szs = [], [], []
        for n in active:
            w = torch.randint(0, 16, (f, d), device=DEV, dtype=torch.int32)
            wu8 = ((w[:, ::2] << 4) | w[:, 1::2]).to(torch.uint8)
            ws.append(torch._convert_weight_to_int4pack(wu8, 8))
            sz = torch.empty(d // 128, f, 2, device=DEV, dtype=torch.bfloat16)
            sz[..., 0] = 0.01
            sz[..., 1] = 0.0
            szs.append(sz)
            xs.append(torch.randn(n, d, device=DEV, dtype=torch.bfloat16))

        def run():
            for x, wp, sz in zip(xs, ws, szs):
                torch._weight_int4pack_mm(x, wp, 128, sz)

        # Capture the per-expert calls in a CUDA Graph and time the replay, so Python and
        # launch overhead between experts is not charged to tinygemm.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        ms = time_cold(graph.replay)
        mb = len(active) * (d * f / 2 + (d // 128) * f * 2 * 2) / 1e6
        print(f"  {f'T={T} E={E} d={d} f={f}':<34}{len(active):>8}{ms:>10.4f}{mb:>11.1f}{mb / ms:>8.1f}")
        del ws, xs, szs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

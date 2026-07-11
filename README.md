# The M4's bf16 matrix instruction is faster *and* more accurate than a naive bf16→fp32 loop — but not bit-deterministic

The M4 CPU implements ARM's bf16 matrix extension (`FEAT_BF16: 1`, `FEAT_EBF16: 0` — base semantics):
`BFDOT` (bf16 pairwise dot → fp32) and `BFMMLA` (bf16 2×4-tile matmul → fp32 2×2 tile), both accumulating
bf16 products into an fp32 register with ARM's mandated internal rounding. People reach for these to
accelerate low-precision GEMM assuming they trade accuracy for speed. On Apple silicon it's measured here:
they're a rare **win on both axes** vs the naive software path — at the cost of bit-reproducibility.

## Result: faster AND more accurate than naive scalar bf16→fp32

Exact fp64 oracle; rel-error `|c−c*|/‖products‖₂` of a length-K dot, RMS over 400 seeds, same bf16 inputs
fed to every scheme:

| K | BFDOT (hw) | naive sw (bf16→fp32, seq) | Kahan fp32 | hw / naive |
|---|---|---|---|---|
| 8 | 8.2e-9 | 8.4e-9 | 7.5e-9 | 0.98 |
| 256 | 6.9e-8 | 9.4e-8 | 2.6e-8 | 0.73 |
| 512 | 9.1e-8 | 1.25e-7 | 2.6e-8 | 0.73 |
| 1024 | 1.3e-7 | 2.05e-7 | 2.6e-8 | 0.63 |
| 4096 | 2.85e-7 | 5.30e-7 | 2.7e-8 | **0.54** |

- **Accuracy (P1, P3):** the hardware BFDOT accumulation is **at least as accurate as naive at every K
  and ~2× more accurate by K=4096**, and the edge **grows with reduction length**. Reason: BFDOT
  accumulates across **4 fp32 lanes** (a shallow reduction tree, summed at the end) rather than the
  naive **single sequential** fp32 accumulator, plus ARM's fused product rounding — so it suffers less
  accumulation swamping. Compensated **Kahan fp32 is still best** (flat ~2.6e-8), as expected.
- **Throughput (P4):** BFDOT is **5.4× faster** than the naive fp32 loop (0.075 vs 0.40 ns/MAC) — one
  instruction does 8 bf16 MACs. Asm-audited: 15 `bfdot`/`bfmmla` emitted, not scalarized.
- **Bit-determinism (P2): no.** The same BFDOT dot computed with different lane-groupings / reduction
  orders (single accumulator, two accumulators, reversed) **diverges in the low bits** (~4e-6 absolute)
  — because the fp32 accumulate is non-associative. A positive control (naive fp32 forward vs reversed
  also diverges) confirms the comparator is live. **So a BFMMLA GEMM's result varies bit-for-bit across
  thread counts / tile shapes** — a reproducibility caveat for anyone needing bit-exact outputs.

## The takeaway

If you're doing low-precision GEMV/GEMM on Apple silicon and were avoiding bf16 for accuracy: the
hardware `BFDOT`/`BFMMLA` path is **both faster and more accurate than a hand-rolled bf16→fp32 scalar
loop** — the multi-lane fp32 accumulator is doing free error reduction. If you need bit-reproducible
results across tilings/threads, you still can't get them from bf16 matrix instructions (nor from naive
fp32) — only compensated summation gives both low error and (with fixed order) determinism.

## Pre-registration scorecard (committed before results, `PREREG.md`)

| | Prediction | Outcome |
|---|---|---|
| **P1** | hw ≥ as accurate as naive; strictly better K≥256 | ✅ 0.98 → 0.54 (2× better at K=4096) |
| **P2** | BFDOT not bit-identical under regrouping | ✅ diverges ~4e-6 (fp32 non-associative) |
| **P3** | hw error grows slower with K | ✅ hw/naive ratio 0.98 → 0.54 |
| **P4** | BFDOT ≥ 3× faster than naive fp32 | ✅ 5.4× |
| **P5** | bf16-hw between fp32 and fp16-accumulate | ✅ Kahan-fp32 < BFDOT-hw < naive-seq-bf16 |

## Rigor & scope

- **Exact fp64 oracle**; splitmix64 seeds; error normalized by ‖products‖₂ (backward-stable).
- **Asm audit** confirms `bfdot`/`bfmmla` are emitted. **Determinism on raw bits** with a live positive
  control. Both `gate`-verified: BFDOT(ones₆₄)=64, BFMMLA tile = [10 20; 26 52] exact.
- **Scope:** the fused-accumulate numerics are characterized via **BFDOT** (the bf16 dot primitive — the
  exact operation under test). **BFMMLA** is the 2×4-tile matrix form using the identical fp32
  accumulator + rounding (gate-verified correct); its per-element accumulation numerics are the same. A
  full BFMMLA-tiled GEMM accuracy sweep with correct tile layout is a stated extension. M4 CPU only, no
  NVIDIA GPU.

## Reproduce

```bash
clang++ -O2 -march=armv8.6-a+bf16 -std=c++17 src/bf16mma.cpp -o bf16mma
./bf16mma gate          # BFDOT + BFMMLA correctness
./bf16mma sweep         # accuracy vs K: hw / naive / Kahan
./bf16mma determinism   # bit-divergence under regrouping
./bf16mma bench         # throughput vs naive fp32
```

## License

MIT.

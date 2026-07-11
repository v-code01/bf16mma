# bf16mma: M4 NEON BFMMLA/BFDOT accuracy and tile determinism vs naive bf16 accumulation

*Pre-registered before any result. Committed first.*

The M4 CPU implements ARM's bf16 matrix extension (`FEAT_BF16: 1`, `FEAT_EBF16: 0` — base semantics
only). `BFMMLA` does a 2×2×4 bf16 matrix-multiply-accumulate into an fp32 tile; `BFDOT` a bf16 pairwise
dot into fp32. ARM's base bf16 mandates a specific rounding of the internal products/sum before the
fp32 accumulate (not IEEE round-to-nearest), designed to reduce double-rounding error. Everyone reaches
for these to accelerate low-precision GEMM, but two things are undocumented for Apple silicon: **(1)
whether the hardware fused bf16 accumulation is more or less accurate than the naive
`bf16→fp32, fp32 multiply, fp32 accumulate` software path, and (2) whether a BFMMLA GEMM is
bit-deterministic under regrouping** (different K-tiling / reduction orders producing identical bits).
Both matter: (1) is a correctness question for anyone swapping in BFMMLA; (2) is a reproducibility
question (bit-exact results across thread counts / tile shapes).

## Exact oracle & method

For a GEMM `C = A·B` (A: M×K, B: K×N) with A,B drawn as bf16-representable values: the **fp64 dot
product** `Σ_k a_ik·b_kj` (products and sum in fp64) is exact ground truth. Contenders, all consuming
the *same* bf16 inputs: **(H) BFMMLA/BFDOT** hardware fused; **(S) naive software** — widen each bf16 to
fp32, fp32 multiply, fp32 sequential accumulate; **(K) Kahan-fp32** compensated software; and **(F16)**
an fp16-accumulate reference (ties the sibling fp16 study). Error metric `|c − c*|/‖products‖₂`
(backward-stable). Determinism: compute the same C via several K-tilings / reduction orders and compare
**raw bit patterns**.

*(Gate at setup: BFDOT/BFMMLA compile with `-march=armv8.6-a+bf16` and execute correctly on the M4.)*

## Predictions (falsifiable)

- **P1 (hardware accuracy).** The BFMMLA/BFDOT fused path is **at least as accurate** as naive software
  bf16→fp32 accumulation over K ∈ {8..4096}: mean rel-error(H) ≤ rel-error(S) at every K, and strictly
  lower for K ≥ 256 (the ARM internal rounding reduces accumulation error). *Falsifier:* H is
  consistently *worse* than S → the hardware fused rounding hurts, not helps.
- **P2 (bit-determinism under regrouping).** A BFMMLA GEMM is **bit-identical** across K-tilings and
  reduction orders (the fp32 accumulate + fixed per-instruction rounding is order-independent up to the
  fp32 add, which is *not* associative) — OR it is not, and we quantify the max bit-divergence.
  Pre-registered direction: BFMMLA regrouping produces **non-bit-identical** results (fp32 accumulation
  is non-associative), with divergence bounded and smaller than naive fp32's. *Falsifier:* bit-identical
  across all regroupings (fully deterministic) — a stronger, cleaner reproducibility result.
- **P3 (error growth vs K).** rel-error(H) grows more slowly with K than naive software bf16 (slope in
  log-log smaller), because the fused rounding is less prone to the sequential-accumulation swamping the
  software path suffers. *Falsifier:* identical slopes → the hardware offers no accumulation-length
  advantage.
- **P4 (throughput).** BFMMLA GEMM throughput (bf16 MAC/s) exceeds the naive fp32 path by ≥ 3× at
  M=N=K=512 (one instruction does 8 bf16 MACs), asm-confirmed. *Falsifier:* < 1.5× → the instruction
  isn't a throughput win on this core.
- **P5 (placement vs fp16/fp32).** bf16-hardware error sits between fp32 (best) and fp16-accumulate
  (worst), quantified — bf16's 8-bit mantissa vs fp16's 10-bit predicts higher per-element error but the
  fused rounding may partly offset it. *Falsifier:* bf16-hardware beats fp32-accumulate (implausible;
  would indicate a measurement bug).

## Independent verification

- **fp64 oracle** cross-checked by a disjoint NumPy fp64 GEMM (< 1e-12 rel).
- **Assembly audit**: confirm `bfmmla`/`bfdot` are actually emitted (not scalarized) and count them.
- **Determinism** measured on raw bits (memcmp), multiple seeds; a positive control (fp32 naive
  regrouping *does* diverge) proves the comparator is live.

## Honest fallbacks

If H is worse than S (P1 false) → the hardware fused rounding is a precision downgrade, a useful warning
for anyone adopting BFMMLA. If BFMMLA is fully bit-deterministic under regrouping (P2 stronger) → a clean
reproducibility guarantee. Every dud is first-class. M4-only, no NVIDIA GPU.

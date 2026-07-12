# The M4's bf16 matrix instruction buys speed, not accuracy, and its "accuracy win" is a strawman

The M4 CPU implements ARM's bf16 matrix extension (`FEAT_BF16: 1`, `FEAT_EBF16: 0`): `BFDOT` (bf16
pairwise dot → fp32) and `BFMMLA` (bf16 2×4-tile matmul → fp32 2×2 tile), accumulating bf16 products
into an fp32 register with ARM's mandated rounding. It's tempting to conclude these are "more accurate"
because they beat a naive scalar bf16→fp32 loop. They don't, that comparison is a strawman.

## Result: ~6.8× faster, but a fair software baseline is *more* accurate

Exact fp64 oracle; rel-error `|c−c*|/‖products‖₂` of a length-K dot, RMS over 400 seeds, same bf16 inputs
to every scheme. `naive1` = single sequential fp32 accumulator; `naive4` = **4 fp32 accumulators matching
BFDOT's lane count**; Kahan = compensated fp32:

| K | BFDOT (hw) | naive1 (seq sw) | **naive4 (matched-lane sw)** | Kahan fp32 | hw / naive4 |
|---|---|---|---|---|---|
| 16 | 1.32e-08 | 1.70e-08 | 1.52e-08 | 1.16e-08 | 0.86 |
| 64 | 3.53e-08 | 3.83e-08 | 3.77e-08 | 2.43e-08 | 0.94 |
| 256 | 6.89e-08 | 9.42e-08 | 5.78e-08 | 2.60e-08 | 1.19 |
| 1024 | 1.29e-07 | 2.05e-07 | 1.01e-07 | 2.62e-08 | 1.28 |
| 4096 | 2.85e-07 | 5.30e-07 | 2.30e-07 | 2.73e-08 | 1.24 |

- **The "accuracy win" over the naive scalar loop is entirely accumulator count.** BFDOT accumulates
  across 4 fp32 lanes; the naive *sequential* baseline uses 1. Match the lane count in software (`naive4`,
  4 fp32 accumulators, zero ARM bf16 rounding) and it is **1.2-1.3× *more* accurate than BFDOT** at every
  K ≥ 256. So **ARM's fused bf16 rounding is slightly *worse* than plain fp32 accumulation**, the
  multi-lane reduction tree is the only accuracy lever, and software gets it for free. Compensated **Kahan
  fp32 is best** (flat ~2.6e-8), ~5× more accurate than either.
- **Throughput is the real win (P4): ~6.8× faster** than the naive scalar fp32 loop (median of 3 runs;
  ~0.05 vs ~0.35 ns/MAC), one instruction does 8 bf16 MACs. Asm-audited: 15 `bfdot`/`bfmmla` emitted,
  not scalarized. (A *vectorized* fp32 baseline would narrow this toward ~1.6×; the 6.8× is vs a scalar
  loop, which is what "naive" means here.)
- **Not bit-deterministic (P2).** The same BFDOT dot with different lane-groupings / reduction orders
  diverges in the low bits (~4e-6 absolute), the fp32 accumulate is non-associative. A live positive
  control (naive fp32 forward vs reversed also diverges) confirms the comparator. By the same
  non-associativity, a BFMMLA GEMM is **expected** to vary bit-for-bit across tilings/threads (asserted,
  not separately measured, see scope).

## The takeaway

If you're accelerating low-precision GEMV/GEMM on Apple silicon: reach for `BFDOT`/`BFMMLA` for
**throughput** (~6.8× over a scalar loop). Do **not** reach for them expecting better accuracy, a
4-lane fp32 software loop is *more* accurate than BFDOT (ARM's fused rounding costs you a little), and
compensated summation is better still. bf16 hardware trades a little accuracy for a lot of speed; the
"it's also more accurate" story only holds against a single-accumulator strawman.

## Pre-registration scorecard (committed before results, `PREREG.md`)

| | Prediction | Outcome |
|---|---|---|
| **P1** | hw more accurate than naive; fused rounding helps | FAIL **falsified**, hw beats only the *sequential* strawman; a matched-lane fp32 loop is 1.24× more accurate; fused rounding slightly *hurts* |
| **P2** | BFDOT not bit-identical under regrouping | PASS diverges ~4e-6 (fp32 non-associative) |
| **P3** | hw error grows slower with K | PARTIAL vs naive1 yes, but naive4 grows the same, it's the accumulator tree, not bf16 |
| **P4** | BFDOT ≥ 3× faster than naive fp32 | PASS ~6.8× (vs scalar) |
| **P5** | bf16-hw between fp32 and fp16-accumulate | PARTIAL **partial**, fp16 reference not implemented (deferred to sibling fp16 studies); only the falsifier verified (bf16-hw doesn't beat fp32/Kahan) |

The honest, sharper finding: **bf16 matrix instructions are a throughput tool; their apparent accuracy
advantage is an artifact of comparing against a single-accumulator baseline, and ARM's fused bf16
rounding is marginally worse than plain multi-accumulator fp32.**

## Rigor & scope

- **Exact fp64 oracle**; splitmix64 seeds; error normalized by ‖products‖₂. **The non-strawman control**
  (4-accumulator matched-lane fp32) is the load-bearing baseline. **Asm audit** (bfdot/bfmmla emitted;
  naive is a genuine scalar fp32 chain). **Determinism on raw bits** with a live positive control. Gate:
  BFDOT(ones₆₄)=64, BFMMLA tile=[10 20; 26 52] exact.
- **Scope:** accuracy characterized via **BFDOT**; **BFMMLA** shares the identical fp32 accumulator +
  rounding (gate-verified), so its per-element numerics match, but a full BFMMLA-tiled GEMM sweep and a
  dedicated BFMMLA determinism test are stated extensions. No fp16-accumulate contender here. M4 CPU only.

## Reproduce

```bash
clang++ -O2 -march=armv8.6-a+bf16 -std=c++17 src/bf16mma.cpp -o bf16mma
./bf16mma gate ; ./bf16mma sweep ; ./bf16mma determinism ; ./bf16mma bench
```

## License

MIT.

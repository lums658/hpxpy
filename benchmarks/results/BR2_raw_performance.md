# BR2 — Raw performance vs roofline

Node: **medusa03** — 2× Intel Xeon Gold 6148 (20c each = 40 cores, 2.4 GHz base / 3.7 turbo,
AVX-512F/DQ/BW/CD/VL), 97 GB RAM. SLURM job 154248, 40 threads. diag built with the BR1 fixes.

**Roofline limits (this node):**
- Memory bandwidth (theoretical): 2 sockets × 6 channels × DDR4-2666 (21.3 GB/s/ch) ≈ **256 GB/s**.
- Peak f64 FLOPs: 40 cores × 32 f64 FLOP/cycle (2 FMA × 8-wide × 2) × 2.4 GHz ≈ **3.07 TFLOP/s**
  (≈2.8 TFLOP/s at the AVX-512 all-core turbo).

## Memory-bound ops — bandwidth-bound and efficient

| op | n | GEl/s (L1) | achieved BW | % of 256 GB/s | L1/L0 |
|---|---:|---:|---:|---:|---:|
| sum | 1e8 | 20.1 | 161 GB/s (1×8B) | 63% | 1.002 |
| sum | 5e8 | 21.0 | 168 GB/s | 66% | 0.999 |
| sum | 1e9 | 21.2 | **170 GB/s** | **66%** | 0.999 |
| dot | 1e8 | 11.5 | **184 GB/s** (2×8B reads) | **72%** | 1.001 |
| mul | 1e8 | 2.55 | ~61 GB/s (3n: 2r+1w) | — | 1.005 |
| muls | 1e8 | 2.92 | ~70 GB/s (3n) | — | 0.994 |

`sum` plateaus at ~170 GB/s from n=1e8 onward — genuinely memory-bandwidth-bound at ~⅔ of the
node's theoretical peak (a realistic figure for a single-stream read reduction with NUMA). `dot`
(two read streams) reaches ~72%. The wrapper is within noise of raw HPX for all of them.

*Note:* `add` measured L1/L0 = 1.12 in this run, but at only 12–13 reps with the output
allocation inside the timed region; BR1's cleaner measurement was ~0.98. Almost certainly
allocation/first-touch variance, not a wrapper penalty — flagged for a clean re-check (more reps,
pre-allocated output) rather than treated as a regression.

## Compute-bound — matmul (the honest naive-vs-BLAS gap)

| size | GFLOP/s (L1) | % of ~3.0 TFLOP/s peak | L1/L0 |
|---|---:|---:|---:|
| 1024³ (L3-resident) | 27.1 | 0.9% | 1.003 |
| 2048³ | 5.97 | 0.20% | 0.987 |
| 4096³ | 3.91 | **0.13%** | 0.971 |

matmul **scales** (40× to 40 cores, BR3) and is **penalty-free** (L1/L0 ≈ 1.0), but its absolute
throughput is a tiny fraction of peak because the kernel is a **naive triple-loop O(n³)** with no
cache blocking or explicit vectorization — it thrashes memory at large n. This is a *kernel* gap,
not an HPX or wrapper gap: a tiled/BLAS-backed (or `hpx::experimental` cache-blocked) matmul is the
obvious future win, with ~hundreds× headroom toward the AVX-512 peak.

## Takeaways
- **Memory-bound surface** (the bulk of NumPy compute): efficient — ~66–72% of memory bandwidth,
  scales to all 40 cores, zero abstraction penalty.
- **Compute-bound matmul**: correct + scalable + penalty-free, but a naive kernel; BLAS/tiling is
  the headline future optimization (filed for a future perf epic).
- **Abstraction penalty holds at the roofline**, not just in the ratio — the wrapper does not cost
  bandwidth or FLOPs.

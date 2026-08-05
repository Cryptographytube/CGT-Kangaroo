# cryptographytube

**Author: sisujhon**

A bounded Pollard's-Kangaroo solver for the secp256k1 elliptic-curve discrete
logarithm problem (ECDLP). Given a compressed public key `P` and a hex interval
`[START, END]`, it recovers the private key `k` such that `k·G == P` **and**
`START ≤ k ≤ END`.

---

## Range confinement

The search is confined to the closed interval `[START, END]` — nothing beyond
`END` is ever produced. This is structural, not a filter:

1. The target is shifted: `P' = P − START·G`, so the relative key
   `kr = k − START` lives in `[0, W)` where `W = END − START + 1`.
2. Tame kangaroos seed in `[0, W)`; wild kangaroos ride on `P'`. Jump sizes are
   `~sqrt(W)`.
3. A trail whose distance exceeds `4*W` is immediately reseeded — it has
   wandered outside the interval and can only encode an out-of-range key.
4. Every candidate must pass two hard checks before printing: `k·G == P` and
   `START ≤ k ≤ END`. A failure prints `REJECTED`, never a solution.

---

## Build

### Windows (one click)

```
build.bat
```

Produces `build\cryptographytube.exe`. The script locates MSVC and nvcc
automatically, then **probes which GPU architectures your CUDA toolkit
supports** and builds a fat binary for all of them.

### Linux / make

```
make                # all supported GPUs
make GPU=native     # only this machine's card (much faster build)
make archinfo       # show which architectures were detected
make selftest       # GPU-vs-CPU field arithmetic test
make clean
```

### GPU support

The build probes each architecture against your nvcc and includes every one it
accepts, plus PTX of the newest for forward compatibility:

| Arch | GPUs |
|---|---|
| sm_52 / sm_61 | Maxwell / Pascal — GTX 750, GTX 10xx *(CUDA ≤ 12 only)* |
| sm_70 / sm_75 | Volta / Turing — RTX 20xx, GTX 16xx |
| sm_80 / sm_86 | Ampere — **RTX 3050, 3060, 3070, 3080, 3090** |
| sm_89 | Ada — **RTX 4050, 4060, 4070, 4080, 4090** |
| sm_90 | Hopper — H100 |
| sm_100 / sm_120 | Blackwell — **RTX 5060, 5070, 5080, 5090** |
| PTX | anything newer, JIT-compiled by the driver |

One exe runs on all of them; no per-GPU rebuild. Older cards are silently
skipped when the installed CUDA no longer supports them (CUDA 13 dropped
Maxwell, Pascal and Volta) — install CUDA 12.x if you need those.

---

## Usage

```
cryptographytube -range START:END -pubkey <hex> [-gpu N] [-dp N] [-herd N]
                 [-kang N] [-step N] [-seed HEX] [-maxops N]
```

| Flag | Meaning |
|---|---|
| `-range START:END` | **Required.** Inclusive hex interval to search. |
| `-pubkey <hex>` | **Required.** Compressed public key (66 hex, `02`/`03`). |
| `-gpu N` | Run on CUDA device `N`. Omit to use the CPU engine. |
| `-dp N` | Distinguished-point bits. Default: auto (`bitlen(W)/2 − 3`). |
| `-herd N` | CPU herd size (default 2048). CPU engine only. |
| `-kang N` | GPU herd size (default: auto, up to 4.2M). |
| `-step N` | GPU walk steps per kernel launch (default 256). |
| `-seed HEX` | PRNG seed — change to vary the walk. |
| `-maxops N` | Stop after `N` group ops without a solution. |

### Example

```
build\cryptographytube.exe -gpu 0 -range 8000000000:FFFFFFFFFF -pubkey 03...
```

---

## Live dashboard

Refreshed twice a second, in place:

```
  CONC: Speed: 2.38 GKeys/s | Ops: 2^34.8 | Time: 0d 00h 00m 12s
  TAMEs: 921 / 398.07T (0.0%) | +73 TAMEs/s
  WILDs: 876 checks | T-W: 0 | W-W: 0 | FP: 0 | 142 DP/s
  DP table: 1.80K stored | dp_bits=24 | herd=4194304 kangaroos
  Last DP: AC241CEFF3F5A9CDB18F39D793A66EF344FD77F5114702CFFF36F1880A000000
  Progress: ~0.0000% of expected 2^73.5 group ops
```

- **CONC** — measured throughput (`ops / elapsed`, not a fixed figure), group
  ops so far, elapsed time
- **TAMEs** — tame DPs found vs expected, and the rate
- **WILDs** — wild DP checks, **T-W** tame/wild collisions, **W-W** same-herd
  collisions, **FP** collisions that failed verification
- **DP table** — stored DPs, DP bit width, herd size
- **Last DP** — x-coordinate of the most recent distinguished point. It changes
  on every refresh while the walk is healthy, so you can see at a glance that
  work is happening. The trailing zero nibbles are the `dp_bits` mask.
- **Progress** — work done against the expected `2^(b/2+1)`

---

## Feasibility

A kangaroo solve needs about `2*sqrt(W)` group operations for width `W`.
At ~2.5 GKeys/s:

| Interval width | Expected work | Wall clock |
|---|---|---|
| 2^40 | 2^21 | instant |
| 2^52 | 2^27 | seconds |
| 2^65 | 2^33.5 | a few seconds |
| 2^80 | 2^41 | ~15 minutes |
| 2^100 | 2^51 | ~10 days |
| 2^145 | 2^73.5 | not feasible on any hardware |

This is a property of the algorithm, not of this implementation. A 145-bit
range will show `Progress: ~0.0000%` indefinitely — that is expected, not a
bug. Use the **Last DP** line to confirm the engine is running.

---

## Design notes

**Field layer.** secp256k1 prime `p = 2^256 − 2^32 − 977`. The 256x256->512
multiply is written in PTX with fused `mad.lo/hi.cc.u64` carry chains, reduced
with the pseudo-Mersenne fold `2^256 ≡ 0x1000003D1 (mod p)`. Inversion uses the
secp256k1 addition chain — 255 squarings + 15 multiplies. Verified bit-for-bit
against the CPU implementation (`make selftest`).

**GPU walk.** Each thread owns 128 kangaroos, sharing a single modular inverse
across the group via Montgomery batch inversion. State is stored SoA
(limb-major) for fully coalesced access. For intervals under 2^120 the kernel
carries 2 distance limbs instead of 4, cutting a third of the per-step traffic.

**Seeding.** Both herds are seeded as arithmetic-progression chains in Jacobian
coordinates, batch-converted to affine with one inverse per block — a
multi-million-kangaroo herd seeds in about a second.

**Batched reseed.** Slots needing a fresh point are staged host-side and pushed
in one `cudaMemcpy` + one kernel call.

The kernel is **memory-bandwidth bound**: at 2.6 GKeys/s it moves ~160 bytes of
state per step, about 420 GB/s of sustained DRAM traffic.

---

## Scope

An ECDLP solver for key recovery within a **known, bounded interval**
(puzzle-solving, CTF, educational use). It only searches the interval you give
it and verifies every result before reporting it.

// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#pragma once
#include "cgt_kangaroo.h"   // Point, U256, SolveResult

namespace cgt {

struct GpuParams {
    int  dp_bits   = -1;     // -1 => auto
    u64  kang       = 0;     // desired herd size (0 => auto from range)
    int  step_cnt   = 256;   // walk steps per kernel launch
    u64  seed       = 0;     // 0 => fixed default
    u64  max_ops    = 0;     // 0 => unlimited
    int  device     = 0;     // CUDA device index
};

// Solve the ECDLP for `target` inside the closed interval [start, end] on the
// GPU. Returns a SolveResult (found/key/group_ops/dp_count). dp_bits actually
// used is written back through `dp_used` when non-null.
SolveResult gpu_solve(const Point& target, const U256& start, const U256& end,
                      const GpuParams& params, int* dp_used = nullptr);

} // namespace cgt

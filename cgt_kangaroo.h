// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#pragma once

#include "cgt_ec.h"

namespace cgt {

struct SolveResult {
    bool  found;
    U256  key;          // absolute private key in [start, end]
    u64   group_ops;    // total point additions performed
    u64   dp_count;     // distinguished points recorded
};

class Kangaroo {
public:
    // target: public key P; interval [start, end] inclusive.
    void configure(const Point& target, const U256& start, const U256& end,
                   int dp_bits = -1, int herd = 1024, u64 seed = 0x9E3779B97F4A7C15ULL);
    SolveResult solve(u64 max_ops = 0);   // max_ops==0 => run until solved

    int  dp_bits()   const { return dp_bits_; }
    const U256& width() const { return W_; }

private:
    Point target_, shifted_;      // P and P' = P - start*G
    U256  start_, end_, W_;       // interval + width
    U256  mean_jump_;             // ~ sqrt(W)/2 (power of two)
    int   dp_bits_ = 20;
    int   herd_ = 1024;
    u64   seed_ = 0;
};

} // namespace cgt

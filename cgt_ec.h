// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#pragma once

#include "cgt_uint.h"

namespace cgt {

struct Point {
    U256 x;
    U256 y;
    bool inf;      // true => point at infinity (group identity)

    void set_infinity()      { x.zero(); y.zero(); inf = true; }
    bool is_infinity() const { return inf; }
    bool equals(const Point& o) const {
        if (inf || o.inf) return inf == o.inf;
        return (x == o.x) && (y == o.y);
    }
    // 33-byte compressed hex (02/03 prefix). Fills y by decompression.
    bool from_compressed_hex(const char* s);
};

// Group / scalar helpers.  All static; no hidden global state beyond the
// generator, which InitCurve() prepares.
struct Curve {
    static void   init();                                  // one-time setup
    static Point  add(const Point& a, const Point& b);     // a + b
    static Point  dbl(const Point& a);                     // 2a
    static Point  neg(const Point& a);                     // -a
    static Point  mul(const U256& k, const Point& base);   // k * base
    static Point  mul_g(const U256& k);                    // k * G
    static bool   decompress(Point& out, const U256& x, bool y_odd);
    static bool   on_curve(const Point& p);
    static const  Point& generator();
};

// scalar arithmetic modulo the group order n (for private-key reconstruction)
namespace sc {
    void add_n(U256& r, const U256& a, const U256& b);  // (a+b) mod n
    void sub_n(U256& r, const U256& a, const U256& b);  // (a-b) mod n
}

} // namespace cgt

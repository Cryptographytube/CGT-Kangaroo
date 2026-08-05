// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_ec.h"

namespace cgt {

static Point G_POINT;   // generator, filled by Curve::init()

void Curve::init() {
    G_POINT.x = { { GEN_X[0], GEN_X[1], GEN_X[2], GEN_X[3] } };
    G_POINT.y = { { GEN_Y[0], GEN_Y[1], GEN_Y[2], GEN_Y[3] } };
    G_POINT.inf = false;
}

const Point& Curve::generator() { return G_POINT; }

Point Curve::neg(const Point& a) {
    Point r = a;
    if (!a.inf) fp::neg(r.y, a.y);
    return r;
}

// affine doubling: lambda = 3x^2 / 2y ; x3 = l^2 - 2x ; y3 = l(x - x3) - y
Point Curve::dbl(const Point& a) {
    Point r;
    if (a.inf || a.y.is_zero()) { r.set_infinity(); return r; }

    U256 x2, num, den, inv_den, lam, t;
    fp::sqr(x2, a.x);                 // x^2
    fp::add(num, x2, x2);             // 2x^2
    fp::add(num, num, x2);            // 3x^2   (a=0 for secp256k1)
    fp::add(den, a.y, a.y);           // 2y
    fp::inv(inv_den, den);
    fp::mul(lam, num, inv_den);       // lambda

    fp::sqr(t, lam);                  // l^2
    fp::sub(t, t, a.x);
    fp::sub(r.x, t, a.x);             // x3 = l^2 - 2x

    fp::sub(t, a.x, r.x);
    fp::mul(t, lam, t);
    fp::sub(r.y, t, a.y);             // y3 = l(x - x3) - y
    r.inf = false;
    return r;
}

// affine addition of distinct points
Point Curve::add(const Point& a, const Point& b) {
    if (a.inf) return b;
    if (b.inf) return a;

    if (a.x == b.x) {
        // same x: either doubling, or opposite y -> infinity
        U256 ny; fp::neg(ny, a.y);
        if (b.y == ny) { Point r; r.set_infinity(); return r; }
        return dbl(a);
    }

    Point r;
    U256 num, den, inv_den, lam, t;
    fp::sub(num, b.y, a.y);           // y2 - y1
    fp::sub(den, b.x, a.x);           // x2 - x1
    fp::inv(inv_den, den);
    fp::mul(lam, num, inv_den);

    fp::sqr(t, lam);
    fp::sub(t, t, a.x);
    fp::sub(r.x, t, b.x);             // x3 = l^2 - x1 - x2

    fp::sub(t, a.x, r.x);
    fp::mul(t, lam, t);
    fp::sub(r.y, t, a.y);             // y3 = l(x1 - x3) - y1
    r.inf = false;
    return r;
}

// left-to-right double-and-add
Point Curve::mul(const U256& k, const Point& base) {
    Point acc; acc.set_infinity();
    for (int i = 255; i >= 0; --i) {
        acc = dbl(acc);
        if (k.bit(i)) acc = add(acc, base);
    }
    return acc;
}

Point Curve::mul_g(const U256& k) { return mul(k, G_POINT); }

bool Curve::on_curve(const Point& p) {
    if (p.inf) return true;
    U256 lhs, rhs, t;
    fp::sqr(lhs, p.y);                // y^2
    fp::sqr(rhs, p.x);
    fp::mul(rhs, rhs, p.x);           // x^3
    t.set(7);
    fp::add(rhs, rhs, t);             // x^3 + 7
    return lhs == rhs;
}

bool Curve::decompress(Point& out, const U256& x, bool y_odd) {
    U256 rhs, t, y;
    fp::sqr(rhs, x);
    fp::mul(rhs, rhs, x);             // x^3
    t.set(7);
    fp::add(rhs, rhs, t);            // x^3 + 7
    if (!fp::sqrt(y, rhs)) return false;
    if (((y.v[0] & 1ULL) != 0) != y_odd) fp::neg(y, y);   // pick requested parity
    out.x.copy(x);
    out.y.copy(y);
    out.inf = false;
    return true;
}

bool Point::from_compressed_hex(const char* s) {
    if (!s) return false;
    if (s[0]=='0' && (s[1]=='x'||s[1]=='X')) s += 2;
    int len = 0; while (s[len]) ++len;
    if (len != 66) return false;                 // 1 prefix byte + 32 x bytes
    bool y_odd;
    if (s[0]=='0' && s[1]=='2') y_odd = false;
    else if (s[0]=='0' && s[1]=='3') y_odd = true;
    else return false;
    U256 xx;
    if (!xx.from_hex(s + 2)) return false;
    return Curve::decompress(*this, xx, y_odd);
}

// --------------------------- scalar mod n ----------------------------------
namespace sc {
    void add_n(U256& r, const U256& a, const U256& b) {
        r.copy(a);
        u8 c = r.add(b);
        if (c || U256::cmp(r, fp::N) >= 0) r.sub(fp::N);
    }
    void sub_n(U256& r, const U256& a, const U256& b) {
        r.copy(a);
        u8 br = r.sub(b);
        if (br) r.add(fp::N);
    }
}

} // namespace cgt

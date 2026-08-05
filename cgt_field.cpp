// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_uint.h"

namespace cgt {
namespace fp {

const U256 P = { { FIELD_P[0], FIELD_P[1], FIELD_P[2], FIELD_P[3] } };
const U256 N = { { GROUP_N[0], GROUP_N[1], GROUP_N[2], GROUP_N[3] } };

// bring a value that is < 2^256 (but possibly >= p) into [0, p)
static inline void cond_sub_p(U256& a) {
    if (U256::cmp(a, P) >= 0) a.sub(P);
}

// reduce a 512-bit little-endian product t[0..7] into r ∈ [0, p)
static void reduce512(const u64 t[8], U256& r) {
    const u64 C = FIELD_C;

    // n = t_high * C  (5 limbs; C is 33 bits so the product spans <= 289 bits)
    u64 n[5];
    u64 carry = 0, hi;
    for (int i = 0; i < 4; ++i) {
        u64 lo = mul_wide(t[4 + i], C, &hi);
        u8  c1 = add_carry(0, lo, carry, &n[i]);
        carry  = hi + c1;                 // hi < 2^33, safe
    }
    n[4] = carry;

    // n[0..3] += t_low, carry ripples into n[4]
    u8 c = 0;
    for (int i = 0; i < 4; ++i) c = add_carry(c, n[i], t[i], &n[i]);
    n[4] += c;

    // fold the small top limb n[4] back in: n[4]*C is < 2^67 (two limbs)
    u64 th, tl = mul_wide(n[4], C, &th);
    u8 cc = 0;
    cc = add_carry(cc, n[0], tl, &r.v[0]);
    cc = add_carry(cc, n[1], th, &r.v[1]);
    cc = add_carry(cc, n[2], 0,  &r.v[2]);
    cc = add_carry(cc, n[3], 0,  &r.v[3]);

    // a leftover carry means one more fold of C
    if (cc) {
        u8 c2 = 0;
        c2 = add_carry(c2, r.v[0], C, &r.v[0]);
        c2 = add_carry(c2, r.v[1], 0, &r.v[1]);
        c2 = add_carry(c2, r.v[2], 0, &r.v[2]);
        c2 = add_carry(c2, r.v[3], 0, &r.v[3]);
    }
    cond_sub_p(r);
}

void add(U256& r, const U256& a, const U256& b) {
    r.copy(a);
    u8 c = r.add(b);
    // if it overflowed 2^256 OR landed >= p, correct by subtracting p
    if (c || U256::cmp(r, P) >= 0) r.sub(P);
}

void sub(U256& r, const U256& a, const U256& b) {
    r.copy(a);
    u8 br = r.sub(b);
    if (br) r.add(P);   // wrapped negative -> add p back
}

void neg(U256& r, const U256& a) {
    if (a.is_zero()) { r.zero(); return; }
    r.copy(P);
    r.sub(a);
}

// schoolbook 256x256 -> 512, then reduce
void mul(U256& r, const U256& a, const U256& b) {
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; ++i) {
        u64 carry = 0;
        for (int j = 0; j < 4; ++j) {
            u64 hi;
            u64 lo = mul_wide(a.v[i], b.v[j], &hi);
            // t[i+j] += lo + carry
            u8 c1 = add_carry(0, t[i + j], lo, &t[i + j]);
            u8 c2 = add_carry(0, t[i + j], carry, &t[i + j]);
            carry = hi + c1 + c2;          // hi < 2^64-2, +2 safe
        }
        t[i + 4] += carry;
    }
    reduce512(t, r);
}

void sqr(U256& r, const U256& a) { mul(r, a, a); }

// modular inverse via Fermat: a^(p-2) mod p. Host-side only (not perf critical).
void inv(U256& r, const U256& a) {
    U256 e;                    // exponent = p - 2
    e.copy(P);
    U256 two; two.set(2);
    e.sub(two);

    U256 result; result.set(1);
    U256 base;   base.copy(a);
    for (int i = 0; i < 256; ++i) {
        if (e.bit(i)) mul(result, result, base);
        sqr(base, base);
    }
    r.copy(result);
}

// modular square root. p ≡ 3 (mod 4) so sqrt(a) = a^((p+1)/4); verify result.
bool sqrt(U256& r, const U256& a) {
    U256 e; e.copy(P);
    e.add_u64(1);
    e.shr(2);                  // (p+1)/4

    U256 result; result.set(1);
    U256 base;   base.copy(a);
    for (int i = 0; i < 256; ++i) {
        if (e.bit(i)) mul(result, result, base);
        sqr(base, base);
    }
    U256 chk; sqr(chk, result);
    if (!(chk == a)) return false;
    r.copy(result);
    return true;
}

} // namespace fp

// --------------------------- hex conversion --------------------------------
static int hexval(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

bool U256::from_hex(const char* s) {
    zero();
    if (!s) return false;
    if (s[0]=='0' && (s[1]=='x'||s[1]=='X')) s += 2;
    int len = 0; while (s[len]) ++len;
    if (len == 0 || len > 64) return false;
    for (int i = 0; i < len; ++i) {
        int d = hexval(s[i]);
        if (d < 0) return false;
        // shift accumulator left by 4 bits then OR the nibble
        shl1(); shl1(); shl1(); shl1();
        v[0] |= (u64)d;
    }
    return true;
}

void U256::to_hex(char out[65]) const {
    static const char* H = "0123456789ABCDEF";
    for (int i = 0; i < 64; ++i) {
        int nib = 63 - i;                 // most-significant nibble first
        u64 limb = v[nib >> 4];
        int shift = (nib & 15) * 4;
        out[i] = H[(limb >> shift) & 0xF];
    }
    out[64] = 0;
}

} // namespace cgt

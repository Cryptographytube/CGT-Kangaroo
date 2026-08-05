// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#pragma once
#include <cstdint>

namespace cgt { namespace gpu {

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned char      u8;

__constant__ u64 D_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ u64 D_C = 0x1000003D1ULL;   // 2^256 mod p

// ---- 64-bit add/sub with carry (used off the hot path) ---------------------
__device__ __forceinline__ u64 addc64(u64 a, u64 b, u32& carry) {
    u64 r = a + b;
    u32 c1 = (r < a);
    u64 r2 = r + carry;
    carry = c1 | (r2 < r);
    return r2;
}
__device__ __forceinline__ u64 subb64(u64 a, u64 b, u32& borrow) {
    u64 r = a - b;
    u32 b1 = (a < b);
    u64 r2 = r - borrow;
    borrow = b1 | (r < borrow);
    return r2;
}
__device__ __forceinline__ u64 mul64(u64 a, u64 b, u64& hi) {
    hi = __umul64hi(a, b);
    return a * b;
}

struct Fp { u64 v[4]; };

__device__ __forceinline__ int cmp(const Fp& a, const Fp& b) {
    #pragma unroll
    for (int i = 3; i >= 0; --i)
        if (a.v[i] != b.v[i]) return a.v[i] < b.v[i] ? -1 : 1;
    return 0;
}

__device__ __forceinline__ void cond_sub_p(Fp& a) {
    bool ge = true, done = false;
    #pragma unroll
    for (int i = 3; i >= 0; --i) if (!done) {
        if (a.v[i] != D_P[i]) { ge = a.v[i] > D_P[i]; done = true; }
    }
    if (!ge) return;
    u32 br = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) a.v[i] = subb64(a.v[i], D_P[i], br);
}

// ---- field add / sub -------------------------------------------------------
__device__ __forceinline__ void fp_add(Fp& r, const Fp& a, const Fp& b) {
    u32 c = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) r.v[i] = addc64(a.v[i], b.v[i], c);
    if (c || cmp(r, *(const Fp*)D_P) >= 0) {
        u32 br = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) r.v[i] = subb64(r.v[i], D_P[i], br);
    }
}

__device__ __forceinline__ void fp_sub(Fp& r, const Fp& a, const Fp& b) {
    u32 br = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) r.v[i] = subb64(a.v[i], b.v[i], br);
    if (br) {
        u32 c = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) r.v[i] = addc64(r.v[i], D_P[i], c);
    }
}

// ---- 256x256 -> 512 schoolbook multiply, PTX mad carry chains --------------
//  Each row i accumulates a[i]*b[0..3] into t[i..i+4]: a lo-product chain
//  (mad.lo.cc / madc.lo.cc) then a hi-product chain (mad.hi.cc / madc.hi.cc),
//  each terminated by an addc that catches the row carry. The full product of
//  two < 2^256 values fits in eight limbs, so the final carry is always zero.
__device__ __forceinline__ void mul_256(u64 t[8], const u64 a[4], const u64 b[4]) {
    asm volatile(
    "{\n\t"
    "mov.u64 %0, 0;  mov.u64 %1, 0;  mov.u64 %2, 0;  mov.u64 %3, 0;\n\t"
    "mov.u64 %4, 0;  mov.u64 %5, 0;  mov.u64 %6, 0;  mov.u64 %7, 0;\n\t"
    // ---- row 0 ----
    "mad.lo.cc.u64  %0, %8,  %12, %0;\n\t"
    "madc.lo.cc.u64 %1, %8,  %13, %1;\n\t"
    "madc.lo.cc.u64 %2, %8,  %14, %2;\n\t"
    "madc.lo.cc.u64 %3, %8,  %15, %3;\n\t"
    "addc.u64       %4, %4, 0;\n\t"
    "mad.hi.cc.u64  %1, %8,  %12, %1;\n\t"
    "madc.hi.cc.u64 %2, %8,  %13, %2;\n\t"
    "madc.hi.cc.u64 %3, %8,  %14, %3;\n\t"
    "madc.hi.cc.u64 %4, %8,  %15, %4;\n\t"
    "addc.u64       %5, %5, 0;\n\t"
    // ---- row 1 ----
    "mad.lo.cc.u64  %1, %9,  %12, %1;\n\t"
    "madc.lo.cc.u64 %2, %9,  %13, %2;\n\t"
    "madc.lo.cc.u64 %3, %9,  %14, %3;\n\t"
    "madc.lo.cc.u64 %4, %9,  %15, %4;\n\t"
    "addc.u64       %5, %5, 0;\n\t"
    "mad.hi.cc.u64  %2, %9,  %12, %2;\n\t"
    "madc.hi.cc.u64 %3, %9,  %13, %3;\n\t"
    "madc.hi.cc.u64 %4, %9,  %14, %4;\n\t"
    "madc.hi.cc.u64 %5, %9,  %15, %5;\n\t"
    "addc.u64       %6, %6, 0;\n\t"
    // ---- row 2 ----
    "mad.lo.cc.u64  %2, %10, %12, %2;\n\t"
    "madc.lo.cc.u64 %3, %10, %13, %3;\n\t"
    "madc.lo.cc.u64 %4, %10, %14, %4;\n\t"
    "madc.lo.cc.u64 %5, %10, %15, %5;\n\t"
    "addc.u64       %6, %6, 0;\n\t"
    "mad.hi.cc.u64  %3, %10, %12, %3;\n\t"
    "madc.hi.cc.u64 %4, %10, %13, %4;\n\t"
    "madc.hi.cc.u64 %5, %10, %14, %5;\n\t"
    "madc.hi.cc.u64 %6, %10, %15, %6;\n\t"
    "addc.u64       %7, %7, 0;\n\t"
    // ---- row 3 ----
    "mad.lo.cc.u64  %3, %11, %12, %3;\n\t"
    "madc.lo.cc.u64 %4, %11, %13, %4;\n\t"
    "madc.lo.cc.u64 %5, %11, %14, %5;\n\t"
    "madc.lo.cc.u64 %6, %11, %15, %6;\n\t"
    "addc.u64       %7, %7, 0;\n\t"
    "mad.hi.cc.u64  %4, %11, %12, %4;\n\t"
    "madc.hi.cc.u64 %5, %11, %13, %5;\n\t"
    "madc.hi.cc.u64 %6, %11, %14, %6;\n\t"
    "madc.hi.cc.u64 %7, %11, %15, %7;\n\t"
    "}\n\t"
    : "=l"(t[0]),"=l"(t[1]),"=l"(t[2]),"=l"(t[3]),
      "=l"(t[4]),"=l"(t[5]),"=l"(t[6]),"=l"(t[7])
    : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
      "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
}

// reduce a 512-bit little-endian product t[0..7] modulo p (pseudo-Mersenne)
__device__ __forceinline__ void reduce512(const u64 t[8], Fp& r) {
    const u64 C = D_C;
    u64 n[5]; u64 carry = 0, hi;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 lo = mul64(t[4 + i], C, hi);
        u32 c1 = 0; u64 s = addc64(lo, carry, c1);
        n[i] = s; carry = hi + c1;
    }
    n[4] = carry;
    u32 c = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) n[i] = addc64(n[i], t[i], c);
    n[4] += c;

    u64 th, tl = mul64(n[4], C, th);
    u32 cc = 0;
    r.v[0] = addc64(n[0], tl, cc);
    r.v[1] = addc64(n[1], th, cc);
    r.v[2] = addc64(n[2], 0,  cc);
    r.v[3] = addc64(n[3], 0,  cc);
    if (cc) {
        u32 c2 = 0;
        r.v[0] = addc64(r.v[0], C, c2);
        r.v[1] = addc64(r.v[1], 0, c2);
        r.v[2] = addc64(r.v[2], 0, c2);
        r.v[3] = addc64(r.v[3], 0, c2);
    }
    cond_sub_p(r);
}

__device__ __forceinline__ void fp_mul(Fp& r, const Fp& a, const Fp& b) {
    u64 t[8]; mul_256(t, a.v, b.v); reduce512(t, r);
}
__device__ __forceinline__ void fp_sqr(Fp& r, const Fp& a) {
    u64 t[8]; mul_256(t, a.v, a.v); reduce512(t, r);
}

// ---- modular inverse: secp256k1 addition chain (exponent p-2) --------------
//  255 squarings + 15 multiplies instead of the ~255 multiplies of a naive
//  bitwise Fermat loop, using the {1,2,3,6,9,11,22,44,88,176,220,223} ladder.
__device__ __forceinline__ void fp_inv(Fp& r, const Fp& a) {
    Fp x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t1;

    fp_sqr(x2, a);        fp_mul(x2, x2, a);        // a^(2^2-1)
    fp_sqr(x3, x2);       fp_mul(x3, x3, a);        // a^(2^3-1)

    x6 = x3;    for (int j=0;j<3; ++j) fp_sqr(x6,x6);     fp_mul(x6,  x6,  x3);
    x9 = x6;    for (int j=0;j<3; ++j) fp_sqr(x9,x9);     fp_mul(x9,  x9,  x3);
    x11 = x9;   for (int j=0;j<2; ++j) fp_sqr(x11,x11);   fp_mul(x11, x11, x2);
    x22 = x11;  for (int j=0;j<11;++j) fp_sqr(x22,x22);   fp_mul(x22, x22, x11);
    x44 = x22;  for (int j=0;j<22;++j) fp_sqr(x44,x44);   fp_mul(x44, x44, x22);
    x88 = x44;  for (int j=0;j<44;++j) fp_sqr(x88,x88);   fp_mul(x88, x88, x44);
    x176 = x88; for (int j=0;j<88;++j) fp_sqr(x176,x176); fp_mul(x176,x176,x88);
    x220 = x176;for (int j=0;j<44;++j) fp_sqr(x220,x220); fp_mul(x220,x220,x44);
    x223 = x220;for (int j=0;j<3; ++j) fp_sqr(x223,x223); fp_mul(x223,x223,x3);

    t1 = x223;
    for (int j=0;j<23;++j) fp_sqr(t1,t1); fp_mul(t1, t1, x22);
    for (int j=0;j<5; ++j) fp_sqr(t1,t1); fp_mul(t1, t1, a);
    for (int j=0;j<3; ++j) fp_sqr(t1,t1); fp_mul(t1, t1, x2);
    for (int j=0;j<2; ++j) fp_sqr(t1,t1); fp_mul(r,  t1, a);
}

} } // namespace cgt::gpu

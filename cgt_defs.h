// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cgt {

// ---- fixed width aliases (project-local, avoid leaking into global scope) ---
using  u8  = std::uint8_t;
using  u16 = std::uint16_t;
using  u32 = std::uint32_t;
using  u64 = std::uint64_t;
using  i64 = std::int64_t;

// A 256-bit value is carried as four 64-bit limbs, limb[0] least-significant.
static constexpr int      LIMBS   = 4;
static constexpr unsigned LIMB_BITS = 64;

// kangaroo herd tags
enum KangType : u8 { TAME = 0, WILD1 = 1, WILD2 = 2 };

// ---------------------------------------------------------------------------
//  Portable wide primitives.
//
//  Under MSVC we lean on the x86-64 intrinsics (_umul128 / _addcarry_u64 /
//  _subborrow_u64). Everywhere else we fall back to the compiler's native
//  128-bit integer. Both paths produce identical results; keeping them behind
//  these three inlines lets the rest of the code stay branch-free and readable.
// ---------------------------------------------------------------------------

// full 64x64 -> 128 multiply; returns low 64, writes high 64 through `hi`.
static inline u64 mul_wide(u64 a, u64 b, u64* hi) {
#if defined(_MSC_VER)
    return _umul128(a, b, hi);
#else
    unsigned __int128 p = (unsigned __int128)a * b;
    *hi = (u64)(p >> 64);
    return (u64)p;
#endif
}

// add-with-carry: out = a + b + carry_in, returns carry_out (0/1).
static inline u8 add_carry(u8 carry_in, u64 a, u64 b, u64* out) {
#if defined(_MSC_VER)
    return _addcarry_u64(carry_in, a, b, out);
#else
    unsigned __int128 s = (unsigned __int128)a + b + carry_in;
    *out = (u64)s;
    return (u8)(s >> 64);
#endif
}

// subtract-with-borrow: out = a - b - borrow_in, returns borrow_out (0/1).
static inline u8 sub_borrow(u8 borrow_in, u64 a, u64 b, u64* out) {
#if defined(_MSC_VER)
    return _subborrow_u64(borrow_in, a, b, out);
#else
    unsigned __int128 d = (unsigned __int128)a - b - borrow_in;
    *out = (u64)d;
    return (u8)((d >> 64) & 1);
#endif
}

// ---------------------------------------------------------------------------
//  secp256k1 domain parameters (little-endian limb order: [0]=LSB .. [3]=MSB)
// ---------------------------------------------------------------------------

// Field prime  p = 2^256 - 2^32 - 977
static constexpr u64 FIELD_P[LIMBS] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

// The folding constant C so that 2^256 ≡ C (mod p):  C = 2^32 + 977 = 0x1000003D1
static constexpr u64 FIELD_C = 0x1000003D1ULL;

// Group order  n
static constexpr u64 GROUP_N[LIMBS] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };

// Generator G
static constexpr u64 GEN_X[LIMBS] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
static constexpr u64 GEN_Y[LIMBS] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

} // namespace cgt

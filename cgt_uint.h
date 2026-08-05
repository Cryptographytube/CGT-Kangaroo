// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#pragma once

#include "cgt_defs.h"
#include <cstring>

namespace cgt {

struct U256 {
    u64 v[LIMBS];

    // -- construction / trivial state ---------------------------------------
    void zero()            { v[0]=v[1]=v[2]=v[3]=0; }
    void set(u64 x)        { v[0]=x; v[1]=v[2]=v[3]=0; }
    void copy(const U256& s){ v[0]=s.v[0]; v[1]=s.v[1]; v[2]=s.v[2]; v[3]=s.v[3]; }

    bool is_zero() const   { return (v[0]|v[1]|v[2]|v[3])==0; }
    bool bit(int i) const  { return (v[i>>6] >> (i&63)) & 1ULL; }

    // -- comparison (unsigned) : returns -1 / 0 / +1 ------------------------
    static int cmp(const U256& a, const U256& b) {
        for (int i = LIMBS-1; i >= 0; --i)
            if (a.v[i] != b.v[i]) return a.v[i] < b.v[i] ? -1 : 1;
        return 0;
    }
    bool operator==(const U256& b) const { return cmp(*this,b)==0; }

    // -- ring arithmetic modulo 2^256 ---------------------------------------
    u8 add(const U256& b) {                       // returns carry-out
        u8 c=0;
        c=add_carry(c, v[0], b.v[0], &v[0]);
        c=add_carry(c, v[1], b.v[1], &v[1]);
        c=add_carry(c, v[2], b.v[2], &v[2]);
        c=add_carry(c, v[3], b.v[3], &v[3]);
        return c;
    }
    u8 sub(const U256& b) {                        // returns borrow-out
        u8 br=0;
        br=sub_borrow(br, v[0], b.v[0], &v[0]);
        br=sub_borrow(br, v[1], b.v[1], &v[1]);
        br=sub_borrow(br, v[2], b.v[2], &v[2]);
        br=sub_borrow(br, v[3], b.v[3], &v[3]);
        return br;
    }
    u8 add_u64(u64 x) {
        u8 c = add_carry(0, v[0], x, &v[0]);
        c = add_carry(c, v[1], 0, &v[1]);
        c = add_carry(c, v[2], 0, &v[2]);
        c = add_carry(c, v[3], 0, &v[3]);
        return c;
    }

    void shr1() {
        v[0] = (v[0]>>1) | (v[1]<<63);
        v[1] = (v[1]>>1) | (v[2]<<63);
        v[2] = (v[2]>>1) | (v[3]<<63);
        v[3] =  v[3]>>1;
    }
    void shl1() {
        v[3] = (v[3]<<1) | (v[2]>>63);
        v[2] = (v[2]<<1) | (v[1]>>63);
        v[1] = (v[1]<<1) | (v[0]>>63);
        v[0] =  v[0]<<1;
    }
    void shr(int n) { while(n-- > 0) shr1(); }     // host-side, clarity over speed
    void shl(int n) { while(n-- > 0) shl1(); }

    // -- hex I/O (big-endian text, 64 chars, no 0x) -------------------------
    bool from_hex(const char* s);
    void to_hex(char out[65]) const;
};

// ---------------------------------------------------------------------------
//  Field layer : arithmetic modulo the secp256k1 prime p.  (cgt_field.cpp)
// ---------------------------------------------------------------------------
namespace fp {
    extern const U256 P;   // the field prime as a U256
    extern const U256 N;   // the group order as a U256

    void add (U256& r, const U256& a, const U256& b);  // (a+b) mod p
    void sub (U256& r, const U256& a, const U256& b);  // (a-b) mod p
    void neg (U256& r, const U256& a);                 // (-a)  mod p
    void mul (U256& r, const U256& a, const U256& b);  // (a*b) mod p
    void sqr (U256& r, const U256& a);                 // (a^2) mod p
    void inv (U256& r, const U256& a);                 // a^(p-2) mod p
    bool sqrt(U256& r, const U256& a);                 // returns false if non-residue
}

} // namespace cgt

// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_ec.h"
#include <cstdio>
#include <cstring>

using namespace cgt;

static int g_fail = 0;
static void check(bool ok, const char* name) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_fail;
}
static bool hex_eq(const U256& a, const char* hex) {
    U256 b; b.from_hex(hex);
    return a == b;
}

int main() {
    Curve::init();
    printf("CGT_Kangaroo self-test (secp256k1 field + group)\n");

    // --- field: inverse round-trip  a * a^-1 == 1 -------------------------
    {
        U256 a; a.from_hex("0123456789ABCDEFFEDCBA98765432100F0F0F0F0F0F0F0FA5A5A5A5A5A5A5A5");
        U256 ai, prod; fp::inv(ai, a); fp::mul(prod, a, ai);
        U256 one; one.set(1);
        check(prod == one, "field inverse round-trip");
    }

    // --- field: sqrt round-trip  (sqrt(a^2))^2 == a^2 ---------------------
    {
        U256 a; a.set(123456789);
        U256 a2, r; fp::sqr(a2, a);
        bool ok = fp::sqrt(r, a2);
        U256 r2; fp::sqr(r2, r);
        check(ok && (r2 == a2), "field sqrt round-trip");
    }

    // --- group: 2G matches the standard doubling of the generator ---------
    {
        Point g2 = Curve::dbl(Curve::generator());
        bool ok = hex_eq(g2.x, "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5")
               && hex_eq(g2.y, "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A");
        check(ok, "2G == known vector");
        check(Curve::on_curve(g2), "2G on curve");
    }

    // --- group: 1*G == G, and via mul_g ----------------------------------
    {
        U256 one; one.set(1);
        Point p = Curve::mul_g(one);
        check(p.equals(Curve::generator()), "1*G == G");
    }

    // --- end-to-end: k*G equals a known compressed public key -------------
    // k = 0x1ABCE5  ->  020C9F7444E8051B17D1DB6EA86BC447F111717D30FDA8BC71E3C862C01A5FB59B
    {
        U256 k; k.from_hex("1ABCE5");
        Point p = Curve::mul_g(k);
        Point pub;
        bool dok = pub.from_compressed_hex(
            "020C9F7444E8051B17D1DB6EA86BC447F111717D30FDA8BC71E3C862C01A5FB59B");
        check(dok, "decompress known pubkey");
        check(p.equals(pub), "k*G == known pubkey (full stack)");
        check(Curve::on_curve(p), "k*G on curve");
    }

    // --- hex round-trip ---------------------------------------------------
    {
        const char* h = "DEADBEEF00112233445566778899AABBCCDDEEFF0123456789ABCDEF0F1E2D3C";
        U256 x; x.from_hex(h);
        char out[65]; x.to_hex(out);
        check(strcmp(out, h) == 0, "hex round-trip");
    }

    // --- scalar mod n: (n-1) + 2 == 1 -------------------------------------
    {
        U256 one; one.set(1);
        U256 two; two.set(2);
        U256 nm1; sc::sub_n(nm1, fp::N, one);   // n-1 mod n
        U256 r;   sc::add_n(r, nm1, two);       // (n-1)+2 = n+1 == 1 mod n
        check(r == one, "scalar (n-1)+2 == 1 mod n");
    }

    printf("%s  (%d failure%s)\n", g_fail ? "SELF-TEST FAILED" : "ALL TESTS PASSED",
           g_fail, g_fail==1?"":"s");
    return g_fail ? 1 : 0;
}

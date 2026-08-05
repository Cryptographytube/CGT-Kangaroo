// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_kangaroo.h"
#include <cstdio>
#include <ctime>

using namespace cgt;

static void show(const char* label, const U256& v) {
    char h[65]; v.to_hex(h);
    // trim leading zeros for readability
    int i = 0; while (i < 63 && h[i] == '0') ++i;
    printf("%s%s\n", label, h + i);
}

int main(int argc, char** argv) {
    Curve::init();

    // interval [start, end] with a deliberately non-power-of-two width
    U256 start, end, key;
    start.from_hex("1000000000");
    end.from_hex("1006FA1B23");         // W = 0x6FA1B24  (~2^27, not a power of two)
    key.from_hex("1004C3D2E1");         // planted key, strictly inside the interval

    if (argc >= 4) { start.from_hex(argv[1]); end.from_hex(argv[2]); key.from_hex(argv[3]); }

    Point pub = Curve::mul_g(key);

    U256 W; fp::sub(W, end, start); W.add_u64(1);
    printf("CGT_Kangaroo — CPU solve test\n");
    show("  start : ", start);
    show("  end   : ", end);
    show("  width : ", W);
    show("  key*  : ", key);              // (only printed so we can grade the result)
    char xh[65]; pub.x.to_hex(xh);
    printf("  pub.x : %s\n", xh);

    Kangaroo solver;
    solver.configure(pub, start, end, /*dp_bits=*/-1, /*herd=*/2048);
    printf("  dp_bits: %d\n\n", solver.dp_bits());

    clock_t t0 = clock();
    SolveResult r = solver.solve(/*max_ops=*/0);
    double secs = double(clock() - t0) / CLOCKS_PER_SEC;

    if (!r.found) { printf("NOT SOLVED\n"); return 1; }

    show("  FOUND : ", r.key);
    bool ok = (r.key == key);
    bool in_range = (U256::cmp(r.key, start) >= 0) && (U256::cmp(r.key, end) <= 0);
    printf("  group_ops : %llu\n", (unsigned long long)r.group_ops);
    printf("  dp_count  : %llu\n", (unsigned long long)r.dp_count);
    printf("  time      : %.3f s\n", secs);
    printf("  correct   : %s\n", ok ? "YES" : "NO");
    printf("  in-range  : %s\n", in_range ? "YES" : "NO");
    printf("%s\n", (ok && in_range) ? "SOLVE TEST PASSED" : "SOLVE TEST FAILED");
    return (ok && in_range) ? 0 : 1;
}

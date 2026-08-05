// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_ec.h"
#include <cstdio>

using namespace cgt;

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: keygen <privkey-hex>\n"); return 2; }
    Curve::init();
    U256 k; if (!k.from_hex(argv[1])) { printf("bad hex\n"); return 2; }
    Point p = Curve::mul_g(k);
    char xh[65]; p.x.to_hex(xh);
    printf("%02d%s\n", (p.y.v[0] & 1) ? 3 : 2, xh);   // 02/03 prefix + x
    return 0;
}

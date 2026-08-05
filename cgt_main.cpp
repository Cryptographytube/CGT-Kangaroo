// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_kangaroo.h"
#include "cgt_gpu.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

using namespace cgt;

static void trim_hex_print(const char* label, const U256& v) {
    char h[65]; v.to_hex(h);
    int i = 0; while (i < 63 && h[i] == '0') ++i;
    printf("%s%s\n", label, h + i);
}

static bool split_range(const char* arg, U256& start, U256& end) {
    const char* colon = strchr(arg, ':');
    if (!colon) return false;
    char lo[80], hi[80];
    size_t nlo = (size_t)(colon - arg);
    if (nlo == 0 || nlo >= sizeof(lo)) return false;
    memcpy(lo, arg, nlo); lo[nlo] = 0;
    strncpy(hi, colon + 1, sizeof(hi) - 1); hi[sizeof(hi) - 1] = 0;
    return start.from_hex(lo) && end.from_hex(hi);
}

static void usage() {
    printf(
      "cryptographytube - bounded ECDLP solver for secp256k1  (author: sisujhon)\n"
      "Usage:\n"
      "  cryptographytube -range START:END -pubkey <hex> [-gpu N] [-dp N] [-herd N]\n"
      "                   [-kang N] [-step N] [-seed HEX] [-maxops N]\n"
      "\n"
      "  -range START:END   inclusive hex interval to search (REQUIRED)\n"
      "  -pubkey <hex>      target compressed public key, 66 hex chars (REQUIRED)\n"
      "  -gpu N             run on CUDA device N (omit => CPU engine)\n"
      "  -dp N              distinguished-point bits (default: auto)\n"
      "  -herd N            CPU kangaroo herd size (default 2048)\n"
      "  -kang N            GPU herd size (default: auto)\n"
      "  -step N            GPU walk steps per launch (default 256)\n"
      "  -seed HEX          PRNG seed (default fixed)\n"
      "  -maxops N          stop after N group ops without a solution\n");
}

int main(int argc, char** argv) {
    Curve::init();

    const char* range_arg = nullptr;
    const char* pub_arg   = nullptr;
    int   dp   = -1;
    int   herd = 2048;
    u64   seed = 0;
    u64   maxops = 0;
    int   use_gpu = -1;      // >=0 => GPU device index
    u64   gpu_kang = 0;      // 0 => auto
    int   gpu_step = 256;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-range")  && i+1 < argc) range_arg = argv[++i];
        else if (!strcmp(argv[i], "-pubkey") && i+1 < argc) pub_arg   = argv[++i];
        else if (!strcmp(argv[i], "-dp")     && i+1 < argc) dp   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-herd")   && i+1 < argc) herd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-maxops") && i+1 < argc) maxops = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-gpu")    && i+1 < argc) use_gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-kang")   && i+1 < argc) gpu_kang = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-step")   && i+1 < argc) gpu_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-seed")   && i+1 < argc) { U256 s; s.from_hex(argv[++i]); seed = s.v[0]; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(); return 0; }
    }

    if (!range_arg || !pub_arg) { usage(); return 2; }

    U256 start, end;
    if (!split_range(range_arg, start, end)) {
        printf("error: -range must be START:END in hex\n"); return 2;
    }
    if (U256::cmp(start, end) > 0) { printf("error: START must be <= END\n"); return 2; }

    Point pub;
    if (!pub.from_compressed_hex(pub_arg)) {
        printf("error: -pubkey must be 66 hex chars (02/03 prefix) on-curve\n"); return 2;
    }

    U256 W; fp::sub(W, end, start); W.add_u64(1);

    printf("======================================================================\n");
    printf(" cryptographytube  -  bounded secp256k1 ECDLP solver   (author: sisujhon)\n");
    printf("======================================================================\n");
    trim_hex_print("  range start : ", start);
    trim_hex_print("  range end   : ", end);
    trim_hex_print("  width       : ", W);
    { char xh[65]; pub.x.to_hex(xh); printf("  target x    : %s\n", xh); }

    Kangaroo solver;
    solver.configure(pub, start, end, dp, herd, seed);

    clock_t t0 = clock();
    SolveResult r{};

    if (use_gpu >= 0) {
        GpuParams gp;
        gp.dp_bits = dp; gp.kang = gpu_kang; gp.step_cnt = gpu_step;
        gp.seed = seed; gp.max_ops = maxops; gp.device = use_gpu;
        int dp_used = -1;
        printf("  engine      : GPU (device %d)\n", use_gpu);
        printf("  search space: ONLY [start, end]  (nothing beyond end)\n\n");
        fflush(stdout);
        r = gpu_solve(pub, start, end, gp, &dp_used);
        printf("  dp_bits     : %d\n", dp_used);
    } else {
        printf("  engine      : CPU\n");
        printf("  dp_bits     : %d\n", solver.dp_bits());
        printf("  herd        : %d\n", herd);
        printf("  search space: ONLY [start, end]  (nothing beyond end)\n\n");
        fflush(stdout);
        r = solver.solve(maxops);
    }

    double secs = double(clock() - t0) / CLOCKS_PER_SEC;

    if (!r.found) {
        printf("NOT SOLVED after %llu group ops (%.2fs)\n",
               (unsigned long long)r.group_ops, secs);
        return 1;
    }

    // hard confinement assertion before we trust the result
    bool in_range = (U256::cmp(r.key, start) >= 0) && (U256::cmp(r.key, end) <= 0);
    Point chk = Curve::mul_g(r.key);
    bool verified = chk.equals(pub);

    trim_hex_print("PRIVATE KEY : ", r.key);
    printf("  verified k*G == P : %s\n", verified ? "YES" : "NO");
    printf("  inside [start,end]: %s\n", in_range ? "YES" : "NO");
    printf("  group_ops         : %llu\n", (unsigned long long)r.group_ops);
    printf("  dp_count          : %llu\n", (unsigned long long)r.dp_count);
    printf("  time              : %.3f s\n", secs);

    if (!verified || !in_range) { printf("REJECTED: result failed confinement/verify\n"); return 1; }
    return 0;
}

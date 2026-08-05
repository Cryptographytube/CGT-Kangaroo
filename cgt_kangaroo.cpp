// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_kangaroo.h"
#include <unordered_map>
#include <vector>
#include <cstdio>

namespace cgt {

// ---- small deterministic PRNG (splitmix64) --------------------------------
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed) {}
    u64 next() {
        u64 z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

static int bit_length(const U256& v) {
    for (int i = 255; i >= 0; --i) if (v.bit(i)) return i + 1;
    return 0;
}

// random value with exactly `nbits` random low bits
static U256 rnd_bits(Rng& rng, int nbits) {
    U256 r; r.zero();
    int limbs = (nbits + 63) / 64;
    for (int i = 0; i < limbs; ++i) r.v[i] = rng.next();
    // mask off the top
    int top = nbits & 63;
    if (top) r.v[limbs - 1] &= ((1ULL << top) - 1);
    for (int i = limbs; i < LIMBS; ++i) r.v[i] = 0;
    return r;
}

// uniform in [0, max) via rejection sampling
static U256 rnd_mod(Rng& rng, const U256& max) {
    int nb = bit_length(max);
    if (nb == 0) { U256 z; z.zero(); return z; }
    for (;;) {
        U256 r = rnd_bits(rng, nb);
        if (U256::cmp(r, max) < 0) return r;
    }
}

// ---- distinguished-point table --------------------------------------------
struct XKey { u64 a, b, c, d; bool operator==(const XKey& o) const {
    return a==o.a && b==o.b && c==o.c && d==o.d; } };
struct XHash { size_t operator()(const XKey& k) const {
    u64 h = k.a*0x100000001B3ULL ^ k.b; h = (h^k.c)*0x100000001B3ULL ^ k.d;
    h ^= h >> 29; return (size_t)h; } };
struct DpEntry { U256 dist; u8 type; };

static const int JMP_CNT_CPU = 32;
static const int JMP_MASK_CPU = JMP_CNT_CPU - 1;

// ---------------------------------------------------------------------------
void Kangaroo::configure(const Point& target, const U256& start, const U256& end,
                         int dp_bits, int herd, u64 seed) {
    target_ = target;
    start_.copy(start);
    end_.copy(end);
    fp::sub(W_, end, start);            // end - start   (values < p so field sub is exact)
    W_.add_u64(1);                      // W = end - start + 1

    // shifted target P' = P - start*G
    Point sg = Curve::mul_g(start_);
    Point neg_sg = Curve::neg(sg);
    shifted_ = Curve::add(target_, neg_sg);

    int b = bit_length(W_);
    int mexp = b / 2 - 1; if (mexp < 1) mexp = 1;
    mean_jump_.zero();
    mean_jump_.v[mexp >> 6] = 1ULL << (mexp & 63);   // 2^mexp ~ sqrt(W)/2

    if (dp_bits < 0) { dp_bits = b / 2 - 3; }
    if (dp_bits < 1) dp_bits = 1;
    if (dp_bits > 24) dp_bits = 24;
    dp_bits_ = dp_bits;
    herd_ = herd < 2 ? 2 : herd;
    seed_ = seed;
}

SolveResult Kangaroo::solve(u64 max_ops) {
    Rng rng(seed_ ? seed_ : 0xDEADBEEFCAFEF00DULL);

    // --- build the shared jump table ---------------------------------------
    std::vector<U256>  jdist(JMP_CNT_CPU);
    std::vector<Point> jpnt(JMP_CNT_CPU);
    U256 half; half.copy(mean_jump_); half.shr1();       // mean/2
    int mbits = bit_length(mean_jump_);
    for (int i = 0; i < JMP_CNT_CPU; ++i) {
        U256 d = rnd_bits(rng, mbits > 0 ? mbits : 1);   // [0, mean)
        d.add(half);                                      // [mean/2, 3mean/2)
        d.v[0] |= 1ULL;                                   // keep it odd
        jdist[i].copy(d);
        jpnt[i] = Curve::mul_g(d);
    }

    // --- kangaroo herd -----------------------------------------------------
    struct Kang { Point pos; U256 dist; u8 type; };
    std::vector<Kang> herd(herd_);
    auto seed_kang = [&](Kang& kg, u8 type) {
        kg.type = type;
        kg.dist = rnd_mod(rng, W_);
        if (type == TAME) {
            kg.pos = Curve::mul_g(kg.dist);
        } else {
            Point off = Curve::mul_g(kg.dist);
            kg.pos = Curve::add(shifted_, off);
        }
    };
    for (int i = 0; i < herd_; ++i) seed_kang(herd[i], (i & 1) ? WILD1 : TAME);

    std::unordered_map<XKey, DpEntry, XHash> table;
    table.reserve(1 << 16);

    const u64 dp_mask = (dp_bits_ >= 64) ? ~0ULL : ((1ULL << dp_bits_) - 1);
    SolveResult res{}; res.found = false;

    auto try_solve = [&](const U256& dt, const U256& dw, Point& probe)->bool {
        // tame pos = dt*G ; wild pos = (kr + dw)*G  =>  kr = dt - dw (mod n)
        U256 kr, k;
        sc::sub_n(kr, dt, dw);
        sc::add_n(k, start_, kr);
        Point chk = Curve::mul_g(k);
        if (chk.equals(target_)) { res.key.copy(k); return true; }
        // fall back to the opposite assignment (guards type bookkeeping)
        sc::sub_n(kr, dw, dt);
        sc::add_n(k, start_, kr);
        chk = Curve::mul_g(k);
        if (chk.equals(target_)) { res.key.copy(k); return true; }
        return false;
    };

    u64 ops = 0, dps = 0;
    for (;;) {
        for (int i = 0; i < herd_; ++i) {
            Kang& kg = herd[i];
            int j = (int)(kg.pos.x.v[0]) & JMP_MASK_CPU;
            kg.pos = Curve::add(kg.pos, jpnt[j]);
            kg.dist.add(jdist[j]);
            ++ops;
            if (kg.pos.inf) { seed_kang(kg, kg.type); continue; }

            if ((kg.pos.x.v[0] & dp_mask) == 0) {           // distinguished point
                ++dps;
                XKey key{ kg.pos.x.v[0], kg.pos.x.v[1], kg.pos.x.v[2], kg.pos.x.v[3] };
                auto it = table.find(key);
                if (it == table.end()) {
                    table.emplace(key, DpEntry{ kg.dist, kg.type });
                } else if (it->second.type != kg.type) {
                    // opposite herds met -> reconstruct
                    U256 dt = (kg.type == TAME) ? kg.dist : it->second.dist;
                    U256 dw = (kg.type == TAME) ? it->second.dist : kg.dist;
                    if (try_solve(dt, dw, kg.pos)) {
                        res.found = true; res.group_ops = ops; res.dp_count = dps;
                        return res;
                    }
                    seed_kang(kg, kg.type);           // false positive -> move on
                } else {
                    // same herd collision (loop / merged trail) -> re-seed
                    if (U256::cmp(it->second.dist, kg.dist) != 0)
                        it->second = DpEntry{ kg.dist, kg.type };
                    seed_kang(kg, kg.type);
                }
            }
            if (max_ops && ops >= max_ops) {
                res.found = false; res.group_ops = ops; res.dp_count = dps;
                return res;
            }
        }
    }
}

} // namespace cgt

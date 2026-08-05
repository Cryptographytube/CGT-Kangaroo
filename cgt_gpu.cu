// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_gpu.h"
#include "cgt_gpu_field.cuh"

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif

namespace cgt {

// ---- tunables --------------------------------------------------------------
//  GRP  : kangaroos per thread. One modular inverse is shared across the whole
//         group (batch inversion), so larger GRP = fewer inverses — but the
//         per-thread dx[]/pre[] scratch grows and starts spilling to local
//         memory, and total thread count (= N/GRP) drops below what fills the
//         SMs. GRP=16 keeps the scratch small and the herd wide.
//  BLOCK: threads per block (multiple of warp size).
static const int GRP      = 128;   // kangaroos per thread (batch-inverse width)
static const int BLOCK    = 256;   // threads per block
static const int JMP_CNT  = 32;    // jump-table entries
static const int JMP_MASK = JMP_CNT - 1;

// ---- device jump table (constant memory) -----------------------------------
__constant__ gpu::u64 cJx[JMP_CNT][4];
__constant__ gpu::u64 cJy[JMP_CNT][4];
__constant__ gpu::u64 cJd[JMP_CNT][4];

struct DP { gpu::u64 x[4]; gpu::u64 d[4]; gpu::u32 type; gpu::u32 slot; };

// ---- device helpers --------------------------------------------------------
using gpu::Fp;
using gpu::u32;
using gpu::u64;

__device__ __forceinline__ Fp ld4(const u64* p) {
    Fp r; r.v[0]=p[0]; r.v[1]=p[1]; r.v[2]=p[2]; r.v[3]=p[3]; return r;
}
__device__ __forceinline__ void st4(u64* p, const Fp& a) {
    p[0]=a.v[0]; p[1]=a.v[1]; p[2]=a.v[2]; p[3]=a.v[3];
}
// SoA (limb-major) load/store: limb l of kangaroo `k` lives at base[l*N + k],
// so consecutive threads touch consecutive addresses -> fully coalesced.
__device__ __forceinline__ Fp ldK(const u64* base, size_t k, size_t N) {
    Fp r; r.v[0]=base[k]; r.v[1]=base[N+k]; r.v[2]=base[2*N+k]; r.v[3]=base[3*N+k]; return r;
}
__device__ __forceinline__ void stK(u64* base, size_t k, size_t N, const Fp& a) {
    base[k]=a.v[0]; base[N+k]=a.v[1]; base[2*N+k]=a.v[2]; base[3*N+k]=a.v[3];
}
__device__ __forceinline__ bool eq4(const Fp& a, const Fp& b) {
    return a.v[0]==b.v[0] && a.v[1]==b.v[1] && a.v[2]==b.v[2] && a.v[3]==b.v[3];
}
// Add a jump distance into a kangaroo's accumulated distance, touching only
// the low DL limbs. Distances never exceed the interval width plus the total
// walked distance, so for narrower intervals the high limbs stay zero and
// reading/writing them is pure wasted bandwidth on the hottest path.
template<int DL>
__device__ __forceinline__ void addDistK(u64* base, size_t k, size_t N, const u64* a) {
    u32 c=0;
    #pragma unroll
    for (int l = 0; l < DL; ++l)
        base[l*N+k] = gpu::addc64(base[l*N+k], a[l], c);
}

// Scatter a batch of freshly-seeded kangaroos into the SoA state arrays.
// Reseeding used to issue a dozen 8-byte cudaMemcpy calls per slot, which on a
// narrow interval (where collisions are frequent) cost far more than the walk
// itself. One kernel over the whole batch removes that entirely.
__global__ void kReseed(u64* Kx, u64* Ky, u64* Kd, size_t N,
                        const u32* idx, const u64* rx, const u64* ry,
                        const u64* rd, int cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;
    size_t slot = idx[i];
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        Kx[l*N+slot] = rx[i*4+l];
        Ky[l*N+slot] = ry[i*4+l];
        Kd[l*N+slot] = rd[i*4+l];
    }
}

// ---- the walk kernel -------------------------------------------------------
//  Kx/Ky/Kd are SoA (limb-major): limb l of kangaroo k at base[l*N + k].
//  Kangaroo index for (thread tid, group g) is  kang = g*T + tid.
//  DL = number of distance limbs the walk actually has to carry (2 or 4).
template<int DL>
__global__ void __launch_bounds__(BLOCK)
             kWalk(u64* Kx, u64* Ky, u64* Kd, const gpu::u8* Kt,
                      int T, size_t N, int steps, u64 dpMask,
                      DP* out, u32* outCnt, u32 outCap) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T) return;

    Fp  pre[GRP];       // prefix products of (jx - x)
    int jj[GRP];        // chosen jump index per kangaroo

    for (int s = 0; s < steps; ++s) {
        // ---- pass 1: denominators + prefix products ----------------------
        Fp acc;
        for (int g = 0; g < GRP; ++g) {
            size_t kang = (size_t)g * T + tid;
            Fp x = ldK(Kx, kang, N);
            int j = (int)(x.v[0]) & JMP_MASK;
            Fp jx = ld4(cJx[j]);
            // self-heal: guarantee jx != x so the affine add is well-defined
            #pragma unroll 1
            for (int t = 0; t < JMP_CNT && eq4(jx, x); ++t) {
                j = (j + 1) & JMP_MASK; jx = ld4(cJx[j]);
            }
            jj[g] = j;
            Fp d; gpu::fp_sub(d, jx, x);
            if (g == 0) acc = d; else gpu::fp_mul(acc, acc, d);
            pre[g] = acc;
        }

        // ---- one inverse for the whole group -----------------------------
        Fp inv; gpu::fp_inv(inv, acc);

        // ---- pass 2: back-substitute, add, accumulate, emit DP -----------
        //  Both dx and x come back from a global re-read + one subtract rather
        //  than a GRP-wide local array: measured faster, because the re-read
        //  hits L2 while a local array of that size does not stay in L1.
        for (int g = GRP - 1; g >= 0; --g) {
            size_t kang = (size_t)g * T + tid;
            int j = jj[g];
            Fp x  = ldK(Kx, kang, N);
            Fp y  = ldK(Ky, kang, N);
            Fp jx = ld4(cJx[j]);
            Fp jy = ld4(cJy[j]);

            Fp d; gpu::fp_sub(d, jx, x);             // recomputed (jx - x)
            Fp invdx;
            if (g == 0) invdx = inv;
            else gpu::fp_mul(invdx, inv, pre[g - 1]);
            gpu::fp_mul(inv, inv, d);                // roll the running inverse

            Fp num; gpu::fp_sub(num, jy, y);         // jy - y
            Fp lam; gpu::fp_mul(lam, num, invdx);    // slope
            Fp lam2; gpu::fp_sqr(lam2, lam);
            Fp x3; gpu::fp_sub(x3, lam2, x); gpu::fp_sub(x3, x3, jx);   // λ² - x - jx
            Fp t;  gpu::fp_sub(t, x, x3); gpu::fp_mul(t, lam, t);
            Fp y3; gpu::fp_sub(y3, t, y);            // λ(x - x3) - y

            stK(Kx, kang, N, x3);
            stK(Ky, kang, N, y3);
            addDistK<DL>(Kd, kang, N, cJd[j]);

            if ((x3.v[0] & dpMask) == 0) {           // distinguished point
                u32 idx = atomicAdd(outCnt, 1u);
                if (idx < outCap) {
                    DP& o = out[idx];
                    o.x[0]=x3.v[0]; o.x[1]=x3.v[1]; o.x[2]=x3.v[2]; o.x[3]=x3.v[3];
                    #pragma unroll
                    for (int l = 0; l < 4; ++l) o.d[l] = (l < DL) ? Kd[l*N+kang] : 0ULL;
                    o.type = Kt[kang];
                    o.slot = (u32)kang;
                }
            }
        }
    }
}

// ===========================================================================
//  Host side
// ===========================================================================

// --- splitmix64 PRNG (matches the CPU solver's generator) ------------------
struct Rng { u64 s; explicit Rng(u64 seed):s(seed){}
    u64 next(){ u64 z=(s+=0x9E3779B97F4A7C15ULL);
        z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
        return z^(z>>31); } };

static int bitlen(const U256& v){ for(int i=255;i>=0;--i) if(v.bit(i)) return i+1; return 0; }

static U256 rnd_bits(Rng& r, int nb){
    U256 x; x.zero(); int limbs=(nb+63)/64;
    for(int i=0;i<limbs;++i) x.v[i]=r.next();
    int top=nb&63; if(top) x.v[limbs-1] &= ((1ULL<<top)-1);
    for(int i=limbs;i<LIMBS;++i) x.v[i]=0; return x;
}
static U256 rnd_mod(Rng& r, const U256& m){
    int nb=bitlen(m); if(!nb){ U256 z; z.zero(); return z; }
    for(;;){ U256 x=rnd_bits(r,nb); if(U256::cmp(x,m)<0) return x; }
}

// --- host DP table ---------------------------------------------------------
struct XKey { u64 a,b,c,d; bool operator==(const XKey& o)const{
    return a==o.a&&b==o.b&&c==o.c&&d==o.d; } };
struct XHash { size_t operator()(const XKey& k)const{
    u64 h=k.a*0x100000001B3ULL^k.b; h=(h^k.c)*0x100000001B3ULL^k.d; h^=h>>29;
    return (size_t)h; } };
struct DpEntry { U256 dist; u8 type; };

static void cu_check(cudaError_t e, const char* what){
    if(e!=cudaSuccess){ printf("CUDA error (%s): %s\n", what, cudaGetErrorString(e)); exit(3); }
}

// human-readable big-number: 1234 -> "1.2K", 1e7 -> "10.0M", 7e13 -> "70.4T"
static void human(char* buf, double v){
    const char* u=""; double x=v;
    if(x>=1e15){ x/=1e15; u="P"; }
    else if(x>=1e12){ x/=1e12; u="T"; }
    else if(x>=1e9){ x/=1e9; u="G"; }
    else if(x>=1e6){ x/=1e6; u="M"; }
    else if(x>=1e3){ x/=1e3; u="K"; }
    if(u[0]) snprintf(buf,16,"%.2f%s",x,u); else snprintf(buf,16,"%.0f",x);
}

// turn on ANSI/VT escape handling so the multi-line dashboard can refresh
// in place on Windows consoles (no-op elsewhere).
static void enable_vt(){
#ifdef _WIN32
    HANDLE h=GetStdHandle(STD_OUTPUT_HANDLE); DWORD m=0;
    if(GetConsoleMode(h,&m)) SetConsoleMode(h, m | 0x0004 /*VIRTUAL_TERMINAL_PROCESSING*/);
#endif
}

SolveResult gpu_solve(const Point& target, const U256& start, const U256& end,
                      const GpuParams& params, int* dp_used) {
    SolveResult res{}; res.found=false;

    cu_check(cudaSetDevice(params.device), "setDevice");

    // ---- interval geometry (identical maths to the CPU configure) ---------
    U256 W; fp::sub(W, end, start); W.add_u64(1);
    int b = bitlen(W);
    int mexp = b/2 - 1; if(mexp<1) mexp=1;
    U256 mean_jump; mean_jump.zero(); mean_jump.v[mexp>>6] = 1ULL<<(mexp&63);

    int dp_bits = params.dp_bits;
    if(dp_bits<0) dp_bits = b/2 - 3;
    if(dp_bits<1) dp_bits=1; if(dp_bits>24) dp_bits=24;
    // A trail covers ~2^dp_bits jumps of ~2^(b/2-1) between distinguished
    // points, i.e. ~2^(dp_bits+b/2-1) of distance. Past dp_bits = b/2 that
    // exceeds the confinement ceiling every single time, so every DP would be
    // retired on arrival and the herd would do nothing but reseed. Clamp it.
    { int dpMax = b/2; if(dpMax<1) dpMax=1; if(dp_bits>dpMax) dp_bits=dpMax; }
    if(dp_used) *dp_used = dp_bits;
    const u64 dpMask = (dp_bits>=64)? ~0ULL : ((1ULL<<dp_bits)-1);

    // shifted target P' = P - start*G
    Point sg = Curve::mul_g(start);
    Point shifted = Curve::add(target, Curve::neg(sg));

    Rng rng(params.seed ? params.seed : 0xDEADBEEFCAFEF00DULL);

    // ---- jump table -------------------------------------------------------
    std::vector<U256> jd(JMP_CNT);
    std::vector<Point> jp(JMP_CNT);
    U256 half; half.copy(mean_jump); half.shr1();
    int mbits = bitlen(mean_jump);
    for(int i=0;i<JMP_CNT;++i){
        U256 d = rnd_bits(rng, mbits>0?mbits:1);
        d.add(half); d.v[0]|=1ULL;
        jd[i].copy(d); jp[i]=Curve::mul_g(d);
    }
    { // upload to constant memory
        std::vector<u64> hx(JMP_CNT*4), hy(JMP_CNT*4), hdd(JMP_CNT*4);
        for(int i=0;i<JMP_CNT;++i){
            for(int l=0;l<4;++l){ hx[i*4+l]=jp[i].x.v[l]; hy[i*4+l]=jp[i].y.v[l]; hdd[i*4+l]=jd[i].v[l]; }
        }
        cu_check(cudaMemcpyToSymbol(cJx, hx.data(), sizeof(u64)*JMP_CNT*4), "cpyJx");
        cu_check(cudaMemcpyToSymbol(cJy, hy.data(), sizeof(u64)*JMP_CNT*4), "cpyJy");
        cu_check(cudaMemcpyToSymbol(cJd, hdd.data(), sizeof(u64)*JMP_CNT*4), "cpyJd");
    }

    // ---- herd sizing ------------------------------------------------------
    //  Scale the default herd to the interval. A kangaroo solve needs only
    //  ~2*sqrt(W) total steps, so a herd of a few * sqrt(W) already saturates
    //  the birthday collision. Requesting the full ~512k herd on a tiny range
    //  wildly oversamples it: trails collide on the first step and the DP
    //  buffer floods, starving the host on reseed churn. Target ~8*sqrt(W)
    //  (2^(b/2+3)), clamped to [8192, 524288]; big ranges still get the max.
    u64 want;
    if (params.kang) {
        want = params.kang;
    } else {
        int hb = b/2 + 3;
        if (hb < 13) hb = 13;          // >= 8192 kangaroos (keep the GPU busy)
        if (hb > 22) hb = 22;          // <= ~4.2M kangaroos (fills all SMs)
        want = 1ULL << hb;
    }
    int T = (int)((want + GRP - 1) / GRP);
    T = ((T + BLOCK - 1) / BLOCK) * BLOCK;         // round up to whole blocks
    if(T < BLOCK) T = BLOCK;
    int blocks = T / BLOCK;
    size_t N = (size_t)T * GRP;

    // ---- seed the herd on the host ---------------------------------------
    //  Two clean arithmetic-progression chains (tame on 0, wild on P'), each
    //  advanced by a fixed stride S*G. To avoid one modular inverse per point,
    //  each chain runs in JACOBIAN coordinates (mixed additions need no inverse)
    //  and is converted to affine in blocks with a SINGLE batched (Montgomery)
    //  inverse per block. Seeding cost drops from N inverses to ~N/B.
    std::vector<u64> hKx(N*4), hKy(N*4), hKd(N*4);
    std::vector<gpu::u8> hKt(N);

    size_t nTame = N / 2;
    size_t nWild = N - nTame;

    // strides tile [0, W): S = max(1, W >> floor(log2(count)))
    auto stride_for = [&](size_t count)->U256 {
        int lg = 0; while (((size_t)1 << (lg+1)) <= count) ++lg;
        U256 s = W; s.shr(lg); if (s.is_zero()) s.set(1); return s;
    };
    U256 St = stride_for(nTame ? nTame : 1);
    U256 Sw = stride_for(nWild ? nWild : 1);
    Point stepT = Curve::mul_g(St);
    Point stepW = Curve::mul_g(Sw);

    U256 d0t = rnd_mod(rng, St);
    U256 d0w = rnd_mod(rng, Sw);
    Point baseT = Curve::mul_g(d0t);
    Point baseW = Curve::add(shifted, Curve::mul_g(d0w));

    // -- local host Jacobian helpers (affine step + running Jacobian point) --
    struct Jac { U256 X, Y, Z; };
    auto j_madd = [&](const Jac& R, const Point& P)->Jac {   // R + P, P affine (Z=1)
        U256 Z1Z1; fp::sqr(Z1Z1, R.Z);
        U256 U2;   fp::mul(U2, P.x, Z1Z1);
        U256 t1;   fp::mul(t1, R.Z, Z1Z1);
        U256 S2;   fp::mul(S2, P.y, t1);
        U256 H;    fp::sub(H, U2, R.X);
        U256 HH;   fp::sqr(HH, H);
        U256 I;    fp::add(I, HH, HH); fp::add(I, I, I);       // 4*HH
        U256 J;    fp::mul(J, H, I);
        U256 r;    fp::sub(r, S2, R.Y); fp::add(r, r, r);      // 2*(S2-Y1)
        U256 V;    fp::mul(V, R.X, I);
        Jac out;
        fp::sqr(out.X, r); fp::sub(out.X, out.X, J);
        U256 twoV; fp::add(twoV, V, V); fp::sub(out.X, out.X, twoV);
        U256 VmX;  fp::sub(VmX, V, out.X); fp::mul(out.Y, r, VmX);
        U256 t2;   fp::mul(t2, R.Y, J); fp::add(t2, t2, t2); fp::sub(out.Y, out.Y, t2);
        U256 ZpH;  fp::add(ZpH, R.Z, H); fp::sqr(out.Z, ZpH);
        fp::sub(out.Z, out.Z, Z1Z1); fp::sub(out.Z, out.Z, HH);
        return out;
    };

    const size_t B = 8192;                          // batch-inversion block
    std::vector<Jac> jbuf(B);
    std::vector<U256> pre(B);

    auto seed_chain = [&](const Point& base, const Point& step, const U256& d0,
                          const U256& S, size_t off, size_t count, u8 type) {
        Jac R{ base.x, base.y, {{1,0,0,0}} };       // Jacobian(base), Z=1
        U256 dist; dist.copy(d0);
        size_t done = 0;
        while (done < count) {
            size_t b = (count - done < B) ? (count - done) : B;
            // fill block, recording distances directly
            for (size_t k = 0; k < b; ++k) {
                jbuf[k] = R;
                size_t idx = off + done + k;
                hKt[idx] = type;
                hKd[0*N+idx]=dist.v[0]; hKd[1*N+idx]=dist.v[1];
                hKd[2*N+idx]=dist.v[2]; hKd[3*N+idx]=dist.v[3];
                R = j_madd(R, step);
                dist.add(S);
            }
            // batched inverse of all Z in the block (Montgomery)
            pre[0].copy(jbuf[0].Z);
            for (size_t k = 1; k < b; ++k) fp::mul(pre[k], pre[k-1], jbuf[k].Z);
            U256 inv; fp::inv(inv, pre[b-1]);
            for (size_t k = b; k-- > 0; ) {
                U256 zinv;
                if (k == 0) zinv.copy(inv);
                else fp::mul(zinv, inv, pre[k-1]);
                U256 tmp; fp::mul(tmp, inv, jbuf[k].Z); inv.copy(tmp);   // roll
                U256 z2, z3, x, y;
                fp::sqr(z2, zinv); fp::mul(z3, z2, zinv);
                fp::mul(x, jbuf[k].X, z2);
                fp::mul(y, jbuf[k].Y, z3);
                size_t idx = off + done + k;
                hKx[0*N+idx]=x.v[0]; hKx[1*N+idx]=x.v[1]; hKx[2*N+idx]=x.v[2]; hKx[3*N+idx]=x.v[3];
                hKy[0*N+idx]=y.v[0]; hKy[1*N+idx]=y.v[1]; hKy[2*N+idx]=y.v[2]; hKy[3*N+idx]=y.v[3];
            }
            done += b;
        }
    };

    seed_chain(baseT, stepT, d0t, St, 0,      nTame, TAME);
    seed_chain(baseW, stepW, d0w, Sw, nTame,  nWild, WILD1);

    // ---- device buffers ---------------------------------------------------
    u64 *dKx,*dKy,*dKd; gpu::u8* dKt;
    cu_check(cudaMalloc(&dKx, N*4*sizeof(u64)), "mKx");
    cu_check(cudaMalloc(&dKy, N*4*sizeof(u64)), "mKy");
    cu_check(cudaMalloc(&dKd, N*4*sizeof(u64)), "mKd");
    cu_check(cudaMalloc(&dKt, N*sizeof(gpu::u8)), "mKt");
    cu_check(cudaMemcpy(dKx,hKx.data(),N*4*sizeof(u64),cudaMemcpyHostToDevice),"cKx");
    cu_check(cudaMemcpy(dKy,hKy.data(),N*4*sizeof(u64),cudaMemcpyHostToDevice),"cKy");
    cu_check(cudaMemcpy(dKd,hKd.data(),N*4*sizeof(u64),cudaMemcpyHostToDevice),"cKd");
    cu_check(cudaMemcpy(dKt,hKt.data(),N*sizeof(gpu::u8),cudaMemcpyHostToDevice),"cKt");

    const u32 DP_CAP = 1u<<20;
    DP* dOut; u32* dCnt;
    cu_check(cudaMalloc(&dOut, (size_t)DP_CAP*sizeof(DP)), "mOut");
    cu_check(cudaMalloc(&dCnt, sizeof(u32)), "mCnt");
    std::vector<DP> hOut(DP_CAP);

    std::unordered_map<XKey,DpEntry,XHash> table;
    table.reserve(1<<20);
    // Hard ceiling on stored DPs (~24M entries, a few GB). Only reached on a
    // run that cannot converge — see the clear() in the launch loop.
    const size_t MAX_TABLE = 24u<<20;

    // ---- reconstruction ---------------------------------------------------
    auto try_solve = [&](const U256& dt, const U256& dw)->bool{
        U256 kr,k;
        sc::sub_n(kr, dt, dw); sc::add_n(k, start, kr);
        if(Curve::mul_g(k).equals(target)){ res.key.copy(k); return true; }
        sc::sub_n(kr, dw, dt); sc::add_n(k, start, kr);
        if(Curve::mul_g(k).equals(target)){ res.key.copy(k); return true; }
        return false;
    };

    // ---- batched reseed ---------------------------------------------------
    //  Slots whose trail collided uselessly get a fresh random position. The
    //  new points are built on the host, staged into packed buffers and pushed
    //  in ONE transfer + ONE kernel, rather than a dozen tiny memcpys per slot.
    std::vector<size_t> reseed;
    std::vector<u32> hRi;                       // slot indices
    std::vector<u64> hRx, hRy, hRd;             // packed x/y/dist (4 limbs each)
    u32 *dRi = nullptr; u64 *dRx=nullptr,*dRy=nullptr,*dRd=nullptr;
    size_t rcap = 0;                            // current device staging capacity

    auto flush_reseed = [&](){
        if (reseed.empty()) return;
        const size_t cnt = reseed.size();
        hRi.resize(cnt); hRx.resize(cnt*4); hRy.resize(cnt*4); hRd.resize(cnt*4);
        for (size_t i = 0; i < cnt; ++i) {
            size_t slot = reseed[i];
            u8 type = hKt[slot];
            U256 dist = rnd_mod(rng, W);
            Point pos = (type==TAME) ? Curve::mul_g(dist)
                                     : Curve::add(shifted, Curve::mul_g(dist));
            hRi[i] = (u32)slot;
            for (int l = 0; l < 4; ++l) {
                hRx[i*4+l] = pos.x.v[l];
                hRy[i*4+l] = pos.y.v[l];
                hRd[i*4+l] = dist.v[l];
                hKx[l*N+slot] = pos.x.v[l];     // keep the host mirror in step
                hKy[l*N+slot] = pos.y.v[l];
                hKd[l*N+slot] = dist.v[l];
            }
        }
        if (cnt > rcap) {                        // grow the staging buffers
            if (dRi) { cudaFree(dRi); cudaFree(dRx); cudaFree(dRy); cudaFree(dRd); }
            rcap = cnt;
            cu_check(cudaMalloc(&dRi, rcap*sizeof(u32)), "mRi");
            cu_check(cudaMalloc(&dRx, rcap*4*sizeof(u64)), "mRx");
            cu_check(cudaMalloc(&dRy, rcap*4*sizeof(u64)), "mRy");
            cu_check(cudaMalloc(&dRd, rcap*4*sizeof(u64)), "mRd");
        }
        cu_check(cudaMemcpy(dRi,hRi.data(),cnt*sizeof(u32),cudaMemcpyHostToDevice),"cRi");
        cu_check(cudaMemcpy(dRx,hRx.data(),cnt*4*sizeof(u64),cudaMemcpyHostToDevice),"cRx");
        cu_check(cudaMemcpy(dRy,hRy.data(),cnt*4*sizeof(u64),cudaMemcpyHostToDevice),"cRy");
        cu_check(cudaMemcpy(dRd,hRd.data(),cnt*4*sizeof(u64),cudaMemcpyHostToDevice),"cRd");
        int rb = (int)((cnt + 255) / 256);
        kReseed<<<rb,256>>>(dKx,dKy,dKd,N,dRi,dRx,dRy,dRd,(int)cnt);
        cu_check(cudaGetLastError(),"reseedLaunch");
    };

    // ---- launch loop ------------------------------------------------------
    const int steps = params.step_cnt>0 ? params.step_cnt : 256;
    // Distance-limb width for the kernel. A kangaroo's accumulated distance is
    // its seed offset (< W) plus the sum of its jumps; jumps average ~sqrt(W),
    // so even an astronomically long run stays far below 2^128 whenever the
    // interval itself is under ~2^120. Carrying 2 limbs instead of 4 removes a
    // third of the per-step global traffic on this bandwidth-bound kernel.
    const int distLimbs = (b <= 120) ? 2 : 4;
    // Distance ceiling for trail confinement. A trail seeded in [0,W) that has
    // travelled more than 4*W is provably useless for a key inside the interval,
    // so it is retired. Skipped when 4*W would overflow 256 bits (b >= 254),
    // where no trail can realistically get there anyway.
    U256 distCap = W;
    bool haveCap = (b <= 253);
    if (haveCap) distCap.shl(2);

    u64 ops=0, dps=0;
    u64 tameSeen=0, wildSeen=0, tw=0, ww=0, fp=0;   // live dashboard counters
    u64 lastDP[4] = {0,0,0,0};                       // last DP x-coordinate for display
    // expected work for a kangaroo solve is ~2*sqrt(W) group ops = ~2^(b/2+1);
    // expected distinguished points ~ that / 2^dp_bits (half tame, half wild).
    double exp_log2  = b/2.0 + 1.0;
    double exp_dp    = std::pow(2.0, exp_log2 - dp_bits);
    double exp_tameDP= exp_dp * 0.5;
    enable_vt();
    auto t0 = std::chrono::steady_clock::now();
    auto tlast = t0;
    bool dash = false;                              // dashboard already drawn?
    for(;;){
        cu_check(cudaMemset(dCnt,0,sizeof(u32)),"memsetCnt");
        if(distLimbs==2)
            kWalk<2><<<blocks,BLOCK>>>(dKx,dKy,dKd,dKt,T,N,steps,dpMask,dOut,dCnt,DP_CAP);
        else
            kWalk<4><<<blocks,BLOCK>>>(dKx,dKy,dKd,dKt,T,N,steps,dpMask,dOut,dCnt,DP_CAP);
        cu_check(cudaGetLastError(),"launch");
        cu_check(cudaDeviceSynchronize(),"sync");
        ops += N * (u64)steps;

        u32 cnt=0; cu_check(cudaMemcpy(&cnt,dCnt,sizeof(u32),cudaMemcpyDeviceToHost),"cCnt");
        u32 got = cnt<DP_CAP ? cnt : DP_CAP;
        if(got){
            cu_check(cudaMemcpy(hOut.data(),dOut,(size_t)got*sizeof(DP),cudaMemcpyDeviceToHost),"cOut");
        }

        reseed.clear();
        // Cap the DP table. A run that cannot succeed (the key simply is not in
        // [start,end]) would otherwise accumulate distinguished points forever
        // and exhaust host memory long before -maxops is reached. Clearing and
        // starting a fresh accumulation costs nothing in correctness: a
        // collision is only useful between two trails that are BOTH still in
        // the table, and dropping old ones just restarts the birthday clock.
        if (table.size() > MAX_TABLE) {
            table.clear();
            table.reserve(1<<20);
        }
        bool solved=false;
        for(u32 i=0;i<got && !solved;++i){
            const DP& o=hOut[i];
            ++dps;
            u8 type=(u8)o.type;
            if(type==TAME) ++tameSeen; else ++wildSeen;
            // Store last DP for dashboard display
            lastDP[0]=o.x[0]; lastDP[1]=o.x[1]; lastDP[2]=o.x[2]; lastDP[3]=o.x[3];
            XKey key{o.x[0],o.x[1],o.x[2],o.x[3]};
            U256 dist; dist.v[0]=o.d[0]; dist.v[1]=o.d[1]; dist.v[2]=o.d[2]; dist.v[3]=o.d[3];
            // Range confinement. A useful tame/wild collision satisfies
            // t - w = kr < W, and both trails start inside [0,W), so neither
            // distance has any reason to grow far past a small multiple of W.
            // A trail that does has wandered off; any collision it makes would
            // encode a key outside [start,end] (that is exactly how an
            // out-of-range key used to surface and get rejected at the very
            // end). Cutting it here keeps the search inside the interval and
            // recycles the slot instead of burning steps beyond END.
            if (haveCap && U256::cmp(dist, distCap) > 0) {
                reseed.push_back(o.slot);
                continue;
            }
            auto it=table.find(key);
            if(it==table.end()){
                table.emplace(key, DpEntry{dist,type});
            } else if(it->second.type != type){
                ++tw;                               // tame/wild collision
                U256 dt = (type==TAME)? dist : it->second.dist;
                U256 dw = (type==TAME)? it->second.dist : dist;
                if(try_solve(dt,dw)){
                    res.found=true; solved=true; break;
                }
                ++fp;                               // collided but didn't solve
                reseed.push_back(o.slot);
            } else {
                ++ww;                               // same-herd collision
                if(U256::cmp(it->second.dist, dist)==0) reseed.push_back(o.slot);
                else it->second = DpEntry{dist,type};
            }
        }
        if(solved){ res.group_ops=ops; res.dp_count=dps; break; }

        flush_reseed();

        // ---- live dashboard (refresh ~2x/sec, in place) -------------------
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - tlast).count();
        if(since >= 0.5){
            double el   = std::chrono::duration<double>(now - t0).count();
            double rate = el>0 ? ops/el : 0;                 // kangaroo steps / s
            double dprate = el>0 ? dps/el : 0;
            double tamerate = el>0 ? tameSeen/el : 0;
            double done_log2 = ops>0 ? std::log2((double)ops) : 0;
            double frac = std::pow(2.0, done_log2 - exp_log2) * 100.0;
            if(frac>100.0) frac=100.0;
            int D=(int)(el/86400), H=((int)el%86400)/3600, M=((int)el%3600)/60, S=(int)el%60;
            char sTame[16],sTameTot[16],sWild[16],sTrate[16],sDrate[16],sTab[16];
            human(sTame,(double)tameSeen); human(sTameTot,exp_tameDP);
            human(sWild,(double)wildSeen); human(sTrate,tamerate); human(sDrate,dprate);
            human(sTab,(double)table.size());
            double tpc = exp_tameDP>0 ? 100.0*tameSeen/exp_tameDP : 0;
            if(tpc>100.0) tpc=100.0;

            if(dash) printf("\033[6A");             // move cursor up over prior block
            printf("\r  CONC: Speed: %.2f GKeys/s | Ops: 2^%.1f | Time: %dd %02dh %02dm %02ds\033[K\n",
                   rate/1e9, done_log2, D,H,M,S);
            printf("\r  TAMEs: %s / %s (%.1f%%) | +%s TAMEs/s\033[K\n", sTame, sTameTot, tpc, sTrate);
            printf("\r  WILDs: %s checks | T-W: %llu | W-W: %llu | FP: %llu | %s DP/s\033[K\n",
                   sWild, (unsigned long long)tw, (unsigned long long)ww, (unsigned long long)fp, sDrate);
            printf("\r  DP table: %s stored | dp_bits=%d | herd=%zu kangaroos\033[K\n",
                   sTab, dp_bits, N);
            printf("\r  Last DP: %016llX%016llX%016llX%016llX\033[K\n",
                   (unsigned long long)lastDP[3], (unsigned long long)lastDP[2],
                   (unsigned long long)lastDP[1], (unsigned long long)lastDP[0]);
            printf("\r  Progress: ~%.4f%% of expected 2^%.1f group ops\033[K\n", frac, exp_log2);
            fflush(stdout);
            dash = true;
            tlast = now;
        }

        if(params.max_ops && ops>=params.max_ops){
            res.found=false; res.group_ops=ops; res.dp_count=dps; break;
        }
    }
    printf("\n");   // finish the dashboard before the result block

    cudaFree(dKx); cudaFree(dKy); cudaFree(dKd); cudaFree(dKt);
    cudaFree(dOut); cudaFree(dCnt);
    if(dRi){ cudaFree(dRi); cudaFree(dRx); cudaFree(dRy); cudaFree(dRd); }
    return res;
}

} // namespace cgt

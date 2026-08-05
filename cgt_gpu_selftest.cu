// ============================================================================
//  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
//  Author: sisujhon
// ============================================================================
#include "cgt_gpu_field.cuh"
#include "cgt_uint.h"          // host oracle (cgt::U256 / cgt::fp)
#include <cstdio>
#include <cstring>

using namespace cgt::gpu;

__global__ void k_ops(const u64* A, const u64* B, u64* MUL, u64* ADD, u64* SUB, u64* INV, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fp a, b, r;
    #pragma unroll
    for (int k = 0; k < 4; ++k) { a.v[k] = A[i*4+k]; b.v[k] = B[i*4+k]; }
    fp_mul(r, a, b); for (int k=0;k<4;++k) MUL[i*4+k]=r.v[k];
    fp_add(r, a, b); for (int k=0;k<4;++k) ADD[i*4+k]=r.v[k];
    fp_sub(r, a, b); for (int k=0;k<4;++k) SUB[i*4+k]=r.v[k];
    fp_inv(r, a);    for (int k=0;k<4;++k) INV[i*4+k]=r.v[k];
}

// simple host PRNG
static cgt::u64 sm(cgt::u64& s){ cgt::u64 z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

int main() {
    const int N = 4096;
    cgt::u64 seed = 0x1234567089ABCDEFULL;

    cgt::U256 *ha = new cgt::U256[N], *hb = new cgt::U256[N];
    for (int i = 0; i < N; ++i) {
        for (int k=0;k<4;++k){ ha[i].v[k]=sm(seed); hb[i].v[k]=sm(seed); }
        // bring into [0,p)
        if (cgt::U256::cmp(ha[i], cgt::fp::P) >= 0) ha[i].sub(cgt::fp::P);
        if (cgt::U256::cmp(hb[i], cgt::fp::P) >= 0) hb[i].sub(cgt::fp::P);
        if (ha[i].is_zero()) ha[i].set(1);
    }

    // host expected
    cgt::U256 *emul=new cgt::U256[N], *eadd=new cgt::U256[N], *esub=new cgt::U256[N], *einv=new cgt::U256[N];
    for (int i=0;i<N;++i){
        cgt::fp::mul(emul[i], ha[i], hb[i]);
        cgt::fp::add(eadd[i], ha[i], hb[i]);
        cgt::fp::sub(esub[i], ha[i], hb[i]);
        cgt::fp::inv(einv[i], ha[i]);
    }

    // device
    u64 *dA,*dB,*dMUL,*dADD,*dSUB,*dINV;
    size_t bytes = (size_t)N*4*sizeof(u64);
    cudaMalloc(&dA,bytes); cudaMalloc(&dB,bytes);
    cudaMalloc(&dMUL,bytes); cudaMalloc(&dADD,bytes); cudaMalloc(&dSUB,bytes); cudaMalloc(&dINV,bytes);
    cudaMemcpy(dA, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hb, bytes, cudaMemcpyHostToDevice);
    k_ops<<<(N+127)/128,128>>>(dA,dB,dMUL,dADD,dSUB,dINV,N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    u64 *gmul=new u64[N*4],*gadd=new u64[N*4],*gsub=new u64[N*4],*ginv=new u64[N*4];
    cudaMemcpy(gmul,dMUL,bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(gadd,dADD,bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(gsub,dSUB,bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(ginv,dINV,bytes,cudaMemcpyDeviceToHost);

    int fails=0;
    auto eq=[&](cgt::U256& e,u64* g){ return memcmp(e.v,g,32)==0; };
    for (int i=0;i<N;++i){
        if(!eq(emul[i],gmul+i*4)) fails++;
        if(!eq(eadd[i],gadd+i*4)) fails++;
        if(!eq(esub[i],gsub+i*4)) fails++;
        if(!eq(einv[i],ginv+i*4)) fails++;
    }
    printf("GPU field self-test: %d operands x4 ops, %d mismatches\n", N, fails);
    printf("%s\n", fails==0 ? "GPU FIELD OK" : "GPU FIELD FAILED");
    return fails==0?0:1;
}

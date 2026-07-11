// bf16mma: does M4 NEON BFDOT/BFMMLA fused bf16 accumulation beat naive bf16->fp32 software, and is
// a BFMMLA GEMM bit-deterministic under regrouping? Exact fp64 oracle.
#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

struct Rng { uint64_t s; Rng(uint64_t x):s(x){}
  uint64_t nx(){ uint64_t z=(s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);}
  float unit(){ return (nx()>>40)*(1.0f/16777216.0f);} };

static inline float bf16_of(float x){ // round f32 -> bf16 (RNE) -> back to f32 value
  uint32_t u; std::memcpy(&u,&x,4); u=(u+0x7FFF+((u>>16)&1))&0xFFFF0000u; float y; std::memcpy(&y,&u,4); return y; }
static inline uint16_t bf16_bits(float x){ uint32_t u; std::memcpy(&u,&x,4); return (uint16_t)((u+0x7FFF+((u>>16)&1))>>16); }

// hardware bf16 dot via BFDOT: acc[f32x4] += sum of bf16 products; a,b are bf16 bit arrays length K (mult of 8)
static float dot_bfdot(const uint16_t* a, const uint16_t* b, int K){
  float32x4_t acc=vdupq_n_f32(0);
  for(int k=0;k<K;k+=8){
    bfloat16x8_t va=vreinterpretq_bf16_u16(vld1q_u16(a+k));
    bfloat16x8_t vb=vreinterpretq_bf16_u16(vld1q_u16(b+k));
    acc=vbfdotq_f32(acc,va,vb);
  }
  return vaddvq_f32(acc);
}
// naive software: widen bf16->fp32, fp32 multiply, fp32 sequential accumulate
static float dot_naive(const float* a,const float* b,int K){ float s=0; for(int k=0;k<K;k++) s+=a[k]*b[k]; return s; }
static float dot_kahan(const float* a,const float* b,int K){ float s=0,c=0; for(int k=0;k<K;k++){ float y=a[k]*b[k]-c; float t=s+y; c=(t-s)-y; s=t;} return s; }
static double dot_f64(const float* a,const float* b,int K){ double s=0; for(int k=0;k<K;k++) s+=(double)a[k]*(double)b[k]; return s; }

// BFDOT with different chunk regrouping (tree vs sequential) for determinism: reduce 4 lanes differently
static float dot_bfdot_regroup(const uint16_t* a,const uint16_t* b,int K,int mode){
  // mode 0: single accumulator (as above). mode 1: two accumulators summed at end. mode 2: reverse order.
  if(mode==1){ float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0);
    for(int k=0;k<K;k+=16){ a0=vbfdotq_f32(a0,vreinterpretq_bf16_u16(vld1q_u16(a+k)),vreinterpretq_bf16_u16(vld1q_u16(b+k)));
      if(k+8<K) a1=vbfdotq_f32(a1,vreinterpretq_bf16_u16(vld1q_u16(a+k+8)),vreinterpretq_bf16_u16(vld1q_u16(b+k+8))); }
    return vaddvq_f32(vaddq_f32(a0,a1)); }
  if(mode==2){ float32x4_t acc=vdupq_n_f32(0);
    for(int k=K-8;k>=0;k-=8) acc=vbfdotq_f32(acc,vreinterpretq_bf16_u16(vld1q_u16(a+k)),vreinterpretq_bf16_u16(vld1q_u16(b+k)));
    return vaddvq_f32(acc); }
  return dot_bfdot(a,b,K);
}

static double relerr(double approx,double orc,double pnorm){ return std::fabs(approx-orc)/(pnorm+1e-30); }



int main(int argc,char**argv){
  std::string cmd=argc>1?argv[1]:"sweep";
  if(cmd=="gate"){
    // BFDOT of ones: K ones dotted with ones = K
    int K=64; std::vector<uint16_t> a(K,bf16_bits(1.0f)),b(K,bf16_bits(1.0f));
    float d=dot_bfdot(a.data(),b.data(),K);
    printf("[gate] BFDOT(ones_64)=%.1f (expect 64)  PASS=%d\n",d,d==64.0f);
    // BFMMLA 2x2 tile sanity: a=[2 rows x 4 cols], b=[2 rows x 4], C[i][j]=row_i(a).row_j(b)
    float av[8]={1,2,3,4, 5,6,7,8}, bv[8]={1,1,1,1, 2,2,2,2};
    uint16_t au[8],bu[8]; for(int i=0;i<8;i++){au[i]=bf16_bits(av[i]);bu[i]=bf16_bits(bv[i]);}
    float32x4_t m=vdupq_n_f32(0); m=vbfmmlaq_f32(m,vreinterpretq_bf16_u16(vld1q_u16(au)),vreinterpretq_bf16_u16(vld1q_u16(bu)));
    // C00=1+2+3+4=10, C01=2+4+6+8=20, C10=5+6+7+8=26, C11=10+12+14+16=52
    printf("[gate] BFMMLA tile C=[%.0f %.0f; %.0f %.0f] (expect 10 20 26 52) PASS=%d\n",
      vgetq_lane_f32(m,0),vgetq_lane_f32(m,1),vgetq_lane_f32(m,2),vgetq_lane_f32(m,3),
      vgetq_lane_f32(m,0)==10&&vgetq_lane_f32(m,1)==20&&vgetq_lane_f32(m,2)==26&&vgetq_lane_f32(m,3)==52);
    return 0;
  }
  if(cmd=="sweep"){
    printf("{\"rows\":[\n");
    int Ks[]={8,16,32,64,128,256,512,1024,2048,4096}; bool first=true;
    for(int K:Ks){
      int R=400; double eH=0,eS=0,eK=0; 
      for(int r=0;r<R;r++){
        Rng g(1000+r*97+K); std::vector<float> af(K),bf(K); std::vector<uint16_t> ab(K),bb(K);
        double s2=0;
        for(int k=0;k<K;k++){ float x=(g.unit()-0.5f)*2, y=(g.unit()-0.5f)*2; af[k]=bf16_of(x); bf[k]=bf16_of(y);
          ab[k]=bf16_bits(x); bb[k]=bf16_bits(y); s2+=(double)af[k]*af[k]*bf[k]*bf[k]; }
        double pnorm=std::sqrt(s2), orc=dot_f64(af.data(),bf.data(),K);
        eH+=std::pow(relerr(dot_bfdot(ab.data(),bb.data(),K),orc,pnorm),2);
        eS+=std::pow(relerr(dot_naive(af.data(),bf.data(),K),orc,pnorm),2);
        eK+=std::pow(relerr(dot_kahan(af.data(),bf.data(),K),orc,pnorm),2);
      }
      printf("%s{\"K\":%d,\"bfdot_hw\":%.4e,\"naive_sw\":%.4e,\"kahan_sw\":%.4e}",first?"":",\n",K,
             std::sqrt(eH/R),std::sqrt(eS/R),std::sqrt(eK/R)); first=false;
    }
    printf("\n]}\n");
    return 0;
  }
  if(cmd=="determinism"){
    // does BFDOT regrouping change the bits?
    int K=4096; Rng g(7); std::vector<uint16_t> a(K),b(K); std::vector<float> af(K),bf(K);
    for(int k=0;k<K;k++){ float x=(g.unit()-0.5f)*2,y=(g.unit()-0.5f)*2; a[k]=bf16_bits(x); b[k]=bf16_bits(y); af[k]=bf16_of(x); bf[k]=bf16_of(y);}
    float d0=dot_bfdot_regroup(a.data(),b.data(),K,0),d1=dot_bfdot_regroup(a.data(),b.data(),K,1),d2=dot_bfdot_regroup(a.data(),b.data(),K,2);
    uint32_t b0,b1,b2; memcpy(&b0,&d0,4);memcpy(&b1,&d1,4);memcpy(&b2,&d2,4);
    // naive fp32 regroup positive control
    float n0=dot_naive(af.data(),bf.data(),K); float nr=0; for(int k=K-1;k>=0;k--) nr+=af[k]*bf[k];
    uint32_t nb0,nb1; memcpy(&nb0,&n0,4);memcpy(&nb1,&nr,4);
    printf("[determinism K=%d] BFDOT single=%.8g 2acc=%.8g reversed=%.8g | bits: single==2acc? %d single==rev? %d\n",K,d0,d1,d2,b0==b1,b0==b2);
    printf("  positive control (naive fp32 fwd vs reversed): bit-equal? %d (should be 0 = comparator live)\n",nb0==nb1);
    return 0;
  }
  if(cmd=="bench"){
    int K=512,reps=200000; Rng g(3); std::vector<uint16_t> a(K),b(K); std::vector<float> af(K),bf(K);
    for(int k=0;k<K;k++){ a[k]=bf16_bits(g.unit()); b[k]=bf16_bits(g.unit()); af[k]=bf16_of(g.unit()); bf[k]=bf16_of(g.unit()); }
    volatile float sink=0;
    auto t0=std::chrono::high_resolution_clock::now(); for(int r=0;r<reps;r++) sink+=dot_bfdot(a.data(),b.data(),K);
    auto t1=std::chrono::high_resolution_clock::now(); for(int r=0;r<reps;r++) sink+=dot_naive(af.data(),bf.data(),K);
    auto t2=std::chrono::high_resolution_clock::now();
    double hw=std::chrono::duration<double,std::nano>(t1-t0).count()/((double)reps*K);
    double sw=std::chrono::duration<double,std::nano>(t2-t1).count()/((double)reps*K);
    printf("[bench K=%d] BFDOT %.4f ns/MAC | naive fp32 %.4f ns/MAC | speedup %.2fx\n",K,hw,sw,sw/hw); (void)sink;
    return 0;
  }
  printf("usage: bf16mma gate|sweep|determinism|bench\n"); return 0;
}

#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>
#include <math.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// LayerNorm over one row of logits (length N): C = ((logit+bias) - mean)/sqrt(var+eps)*gamma + beta.
static inline void layernorm_row(const float* logit, const float* bias,
                                 const float* gamma, const float* beta,
                                 float eps, float* dst, uint32_t N) {
  float mean = 0.0f;
  for (uint32_t j = 0; j < N; ++j) mean += logit[j] + bias[j];
  mean /= (float)N;
  float var = 0.0f;
  for (uint32_t j = 0; j < N; ++j) { float d = (logit[j] + bias[j]) - mean; var += d * d; }
  var /= (float)N;
  float inv = 1.0f / sqrtf(var + eps);
  for (uint32_t j = 0; j < N; ++j) {
    float x = logit[j] + bias[j];
    dst[j] = (x - mean) * inv * gamma[j] + beta[j];
  }
}

// Fused matmul + bias + LayerNorm. See common.h for the three modes.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t N = arg->N, K = arg->K;
  float eps = arg->eps;
  uint32_t tid = threadIdx.x;
  auto pBias = reinterpret_cast<ctx::output_t*>(arg->bias_addr);
  auto pGam  = reinterpret_cast<ctx::output_t*>(arg->ln_w_addr);
  auto pBeta = reinterpret_cast<ctx::output_t*>(arg->ln_b_addr);
  auto pC    = reinterpret_cast<ctx::output_t*>(arg->C_addr);

  if (arg->mode == 0) {
    auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    uint32_t tile_row = blockIdx.x * ctx::tileM;
    auto strip = reinterpret_cast<ctx::output_t*>(__local_mem()); // [tileM x N]

    for (uint32_t cblk = 0; cblk < N; cblk += ctx::tileN) {
      ctx::fragment_a fragA; ctx::fragment_b fragB; ctx::fragment_acc fragC;
      ctx::fill_fragment(fragC, 0);
      for (uint32_t i = 0; i < K; i += ctx::tileK) {
        ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);
        ctx::load_matrix_sync<vt::col_major>(fragB, pB + cblk * K + i, K);
        ctx::mma_sync(fragC, fragA, fragB, fragC);
      }
      ctx::store_matrix_sync(strip + cblk, fragC, N);
      __syncthreads();
    }
    if (tid < ctx::tileM)
      layernorm_row(strip + tid * N, pBias, pGam, pBeta, eps, pC + (tile_row + tid) * N, N);
  } else if (arg->mode == 1) {
    if (vx_thread_id() == 0) {
      dtensor_start(arg->desc_addr);
      while (0 == dtensor_poll()) {}
    }
  } else {
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);
    uint32_t tile_row = blockIdx.x * ctx::tileM;
    if (tid < ctx::tileM)
      layernorm_row(pD + (tile_row + tid) * N, pBias, pGam, pBeta, eps, pC + (tile_row + tid) * N, N);
  }
}

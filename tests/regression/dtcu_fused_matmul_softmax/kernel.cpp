#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>
#include <math.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// Attention-style scale so the pre-softmax logits stay well-conditioned (K=256).
#define SM_SCALE 0.0625f  // 1/sqrt(256)

// Row-wise softmax over the scaled logits held in `row` (length N), writing to `dst`.
// expf is soft-float and dominates the epilogue, so compute it once per element (cache
// into dst) then normalize in place -- half the expf calls of a two-pass formulation.
static inline void softmax_row(const float* row, float* dst, uint32_t N) {
  float m = -INFINITY;
  for (uint32_t j = 0; j < N; ++j) { float v = row[j] * SM_SCALE; if (v > m) m = v; }
  float s = 0.0f;
  for (uint32_t j = 0; j < N; ++j) { float e = expf(row[j] * SM_SCALE - m); dst[j] = e; s += e; }
  float inv = 1.0f / s;
  for (uint32_t j = 0; j < N; ++j) dst[j] *= inv;
}

// Fused matmul + softmax. See common.h for the three modes.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t N = arg->N, K = arg->K;
  uint32_t tid = threadIdx.x;

  if (arg->mode == 0) {
    // In-core fused: build the whole TM x N logit strip in local memory, softmax on-chip.
    auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
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
      ctx::store_matrix_sync(strip + cblk, fragC, N); // tile at column cblk, ldm=N
      __syncthreads();
    }
    if (tid < ctx::tileM)
      softmax_row(strip + tid * N, pC + (tile_row + tid) * N, N);
  } else if (arg->mode == 1) {
    if (vx_thread_id() == 0) {
      dtensor_start(arg->desc_addr);
      while (0 == dtensor_poll()) {}
    }
  } else {
    // SIMT epilogue over the DTCU logits in global memory.
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    uint32_t tile_row = blockIdx.x * ctx::tileM;
    if (tid < ctx::tileM)
      softmax_row(pD + (tile_row + tid) * N, pC + (tile_row + tid) * N, N);
  }
}

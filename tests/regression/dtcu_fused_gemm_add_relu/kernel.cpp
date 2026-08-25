#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// Fused GEMM + bias + ReLU. See common.h for the three modes.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t tid = threadIdx.x;

  if (arg->mode == 0) {
    // In-core fused: matmul this tile, keep it in local memory, apply bias+relu.
    auto pA    = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB    = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pBias = reinterpret_cast<ctx::output_t*>(arg->bias_addr);
    auto pC    = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fragment_a fragA; ctx::fragment_b fragB; ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, 0);
    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);
      ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
    }
    auto local_C = reinterpret_cast<ctx::output_t*>(__local_mem());
    ctx::store_matrix_sync(local_C, fragC, ctx::tileN);
    __syncthreads();
    if (tid < ctx::tileM) {
      for (uint32_t c = 0; c < ctx::tileN; ++c) {
        float v = local_C[tid * ctx::tileN + c] + pBias[tile_col + c];
        pC[(tile_row + tid) * N + tile_col + c] = (v > 0.0f) ? v : 0.0f;
      }
    }
  } else if (arg->mode == 1) {
    // DTCU matmul: D = A*B, written to global memory.
    if (vx_thread_id() == 0) {
      dtensor_start(arg->desc_addr);
      while (0 == dtensor_poll()) {}
    }
  } else {
    // SIMT epilogue over the DTCU output in global memory: C = relu(D + bias).
    auto pD    = reinterpret_cast<ctx::output_t*>(arg->D_addr);
    auto pBias = reinterpret_cast<ctx::output_t*>(arg->bias_addr);
    auto pC    = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;
    if (tid < ctx::tileM) {
      for (uint32_t c = 0; c < ctx::tileN; ++c) {
        float v = pD[(tile_row + tid) * N + tile_col + c] + pBias[tile_col + c];
        pC[(tile_row + tid) * N + tile_col + c] = (v > 0.0f) ? v : 0.0f;
      }
    }
  }
}

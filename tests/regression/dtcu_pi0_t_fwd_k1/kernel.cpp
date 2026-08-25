#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// dtcu_pi0_i_k1: one kernel binary, two modes selected by arg->mode.
//   mode 0 -> in-core TCU. A block computes one tileM x tileN output tile:
//             zero-accumulate matmul, store the tile to local memory, then a SIMT
//             epilogue adds bias[col] and writes D. This mirrors the bench pi0_k1
//             kernel and avoids load_matrix_sync into an accumulator (broken at NT>4).
//             Launched as a (N/tileN, M/tileM) grid of NUM_THREADS-wide blocks.
//   mode 1 -> DTCU. A single leader thread fires the whole-GEMM descriptor
//             (D = C + A*B with C seeded to the bias broadcast) and spins on poll.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  if (arg->mode == 0) {
    auto pA    = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB    = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pD    = reinterpret_cast<ctx::output_t*>(arg->D_addr);
    auto pBias = reinterpret_cast<ctx::output_t*>(arg->bias_addr);

    uint32_t N = arg->N;
    uint32_t K = arg->K;

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragC;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    ctx::fill_fragment(fragC, 0);
    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      auto pTileA = pA + tile_row * K + i;
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync(fragA, pTileA, K);              // A row-major
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K); // B col-major
      ctx::mma_sync(fragC, fragA, fragB, fragC);
    }

    // Spill the A*B tile to local memory, then add bias in a SIMT epilogue.
    auto local_C = reinterpret_cast<ctx::output_t*>(__local_mem());
    ctx::store_matrix_sync(local_C, fragC, ctx::tileN);
    __syncthreads();

    uint32_t tid = threadIdx.x;
    if (tid < ctx::tileM) {
      for (uint32_t c = 0; c < ctx::tileN; ++c) {
        float v = local_C[tid * ctx::tileN + c] + pBias[tile_col + c];
        pD[(tile_row + tid) * N + tile_col + c] = v;
      }
    }
  } else {
    if (vx_thread_id() == 0) {
      dtensor_start(arg->desc_addr);
      while (0 == dtensor_poll()) {
        // busy-wait until the DTCU signals completion
      }
    }
  }
}

// dtcu_edge/incore: GEMM on the in-core TCU, then SIMT scales the output IN REGISTERS (fused) and stores.
// The GEMM result never leaves the register file before the post-processing, so there is no
// output reload from memory. kernel-body cycles are bracketed with MCYCLE (one stamp per warp).

#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint64_t t0 = csr_read(VX_CSR_MCYCLE);

  const uint32_t N = arg->N;
  const uint32_t K = arg->K;
  const uint32_t tile_row = blockIdx.y * ctx::tileM;
  const uint32_t tile_col = blockIdx.x * ctx::tileN;
  const uint32_t warp_gid = blockIdx.y * gridDim.x + blockIdx.x;

  auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragD;
  ctx::load_matrix_sync(fragD, pC + tile_row * N + tile_col, N);
  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);
    ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K);
    ctx::mma_sync(fragD, fragA, fragB, fragD);
  }
  // fused post-processing: scale the accumulator in registers, no memory round-trip
  const float s = arg->scale;
  for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
    fragD.data[r] *= s;
  ctx::store_matrix_sync(pD + tile_row * N + tile_col, fragD, N);

  uint64_t t1 = csr_read(VX_CSR_MCYCLE);
  if (vx_thread_id() == 0) {
    uint64_t* cyc = reinterpret_cast<uint64_t*>(arg->cyc_addr);
    cyc[warp_gid] = t0;
    cyc[arg->num_warps_total + warp_gid] = t1;
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}

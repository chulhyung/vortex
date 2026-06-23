// dtcu_edge/dtcu: GEMM offloaded to the DTCU engine (-> output D in memory), then SIMT RELOADS
// each output tile from memory, scales it, and stores back. The result must travel memory -> core RF
// -> memory (the non-fused path). block(0,0) fires the engine; every block polls done (no barrier).

#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint64_t t0 = csr_read(VX_CSR_MCYCLE);

  const uint32_t N = arg->N;
  const uint32_t tile_row = blockIdx.y * ctx::tileM;
  const uint32_t tile_col = blockIdx.x * ctx::tileN;
  const uint32_t warp_gid = blockIdx.y * gridDim.x + blockIdx.x;
  const float s = arg->scale;
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  // block(0,0) fires the whole-GEMM once; every block's lane 0 polls the done bit
  if (vx_thread_id() == 0) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && vx_warp_id() == 0)
      vt::dtensor_start(arg->desc_addr);
    while (0 == vt::dtensor_poll()) { /* wait for DTCU GEMM */ }
  }
  // non-fused post-processing: reload this tile from memory, scale, store back
  for (uint32_t e = threadIdx.x; e < ctx::tileM * ctx::tileN; e += blockDim.x) {
    uint32_t r = e / ctx::tileN;
    uint32_t c = e % ctx::tileN;
    uint32_t idx = (tile_row + r) * N + (tile_col + c);
    pD[idx] = pD[idx] * s;
  }

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

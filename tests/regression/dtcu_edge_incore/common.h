#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 16
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// GEMM size = SIZE_MULT * native tile. DTCU tile 64x32x16(fp16); in-core(NT=16) tile 16x8x16.
#ifndef SIZE_MULT
#define SIZE_MULT 2
#endif

// SIMT post-processing: out = (C + A*B) * POST_SCALE. 0.5 is exact in fp32 (no rounding drift).
#ifndef POST_SCALE
#define POST_SCALE 0.5f
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t D_addr;
  uint64_t desc_addr;       // DTCU test only (unused by in-core)
  uint64_t cyc_addr;        // 2*num_warps_total u64: [start...][end...] per-warp MCYCLE stamps
  float    scale;           // post-proc multiplier
  uint32_t num_warps_total; // grid_x*grid_y (one warp per block)
} kernel_arg_t;

#endif

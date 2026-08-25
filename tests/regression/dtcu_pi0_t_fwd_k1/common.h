#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Ported from bench robotics_training/pi0/thor/fwd/pi0_fwd_k1 (Pi0 fwd linear, bf16 WMMA).
// GEMM+bias: D = A*B + bias, run two ways -- in-core TCU (mode 0) vs DTCU (mode 1).
// In-core uses zero-accumulator + SIMT bias epilogue (off the NT>4 accumulator-load path);
// DTCU folds bias into its C accumulator (C = bias broadcast, flags=0 -> D = C + A*B).

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#ifndef ITYPE
#define ITYPE bf16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t mode;        // 0: in-core TCU, 1: DTCU
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;      // DTCU: [M*N] bias broadcast accumulator
  uint64_t D_addr;      // output [M*N]
  uint64_t bias_addr;   // in-core: [N] bias vector
  uint64_t desc_addr;
} kernel_arg_t;

#endif

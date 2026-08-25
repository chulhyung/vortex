#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Ported from bench ml_kernels/fused/kbt_matmul_softmax: C = softmax(A*B) row-wise.
// In-core fused keeps each TM-row strip (TM x N) in local memory and reduces on-chip.
// The DTCU path writes A*B to global memory (mode 1) and a SIMT pass reads the full
// rows back to run softmax (mode 2) -- the round-trip is the hand-off overhead.
//   mode 0: in-core fused  -> C = softmax(A*B)   (1D row-tile grid, strip in local mem)
//   mode 1: DTCU matmul    -> D = A*B
//   mode 2: SIMT epilogue  -> C = softmax(D)      (1D row-tile grid, reads global D)

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t mode;
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t D_addr;      // A*B intermediate (DTCU path)
  uint64_t C_addr;      // final output [M*N]
  uint64_t desc_addr;
} kernel_arg_t;

#endif
